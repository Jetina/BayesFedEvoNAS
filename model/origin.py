# -*- coding: utf-8 -*-
# import logging
# import time
# import numpy as np
# import torch
# from torch import nn
# from thop import profile
#
# from darts import genotypes
# from darts.model import NetworkCIFAR
# from darts.model_search import Network
import logging
import time
import numpy as np
import torch
from torch import nn
from thop import profile

from darts import genotypes
from darts.model import NetworkCIFAR
from darts.model_search import Network


class FedNASAggregator(object):


    def __init__(self, train_global, test_global, all_train_data_num, client_num, device, args):
        self.train_global = train_global
        self.test_global = test_global
        self.all_train_data_num = all_train_data_num
        self.client_num = client_num
        self.device = device
        self.args = args

        self.model = self.init_model()


        self.model_dict = {}
        self.arch_dict = {}
        self.sample_num_dict = {}


        self.train_acc_dict = {}
        self.train_loss_dict = {}
        self.train_acc_avg = 0.0
        self.train_loss_avg = 0.0
        self.test_acc_avg = 0.0
        self.test_loss_avg = 0.0

        self.predicted_accs_dict = {}
        self.flag_client_predicted_accs_uploaded_dict = {i: False for i in range(self.client_num)}

        self.flag_client_model_uploaded_dict = {i: False for i in range(self.client_num)}

        self.best_accuracy = 0.0
        self.best_accuracy_different_cnn_counts = {}

    def init_model(self):
        criterion = nn.CrossEntropyLoss().to(self.device)
        if self.args.stage == "search":
            model = Network(self.args.init_channels, 10, self.args.layers, criterion, self.device)
        else:
            genotype = genotypes.FedNAS_V1
            logging.info(genotype)
            model = NetworkCIFAR(self.args.init_channels, 10, self.args.layers, self.args.auxiliary, genotype)
        model.to(self.device)
        return model

    def get_model(self):
        return self.model


    def _get_arch_param_ids(self, model):
        if hasattr(model, "arch_parameters"):
            try:
                return set(id(p) for p in model.arch_parameters())
            except Exception:
                return set()
        return set()

    def _count_params(self, model):
        arch_ids = self._get_arch_param_ids(model)

        total_params = 0
        weight_params = 0
        arch_params = 0
        trainable_params = 0
        params_no_auxiliary = 0

        for name, p in model.named_parameters():
            n = p.numel()

            total_params += n

            if p.requires_grad:
                trainable_params += n

            if id(p) in arch_ids:
                arch_params += n
            else:
                weight_params += n

            if "auxiliary" not in name:
                params_no_auxiliary += n

        return {
            "total_params": total_params,
            "weight_params": weight_params,
            "arch_params": arch_params,
            "trainable_params": trainable_params,
            "params_no_auxiliary": params_no_auxiliary,
        }

    def _calc_global_flops_params(self, input_size=(1, 3, 32, 32)):


        class OnlyLogitsWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                out = self.model(x)
                if isinstance(out, tuple):
                    return out[0]
                return out

        self.model.to(self.device)

        was_training = self.model.training
        self.model.eval()

        dummy_input = torch.randn(*input_size).to(self.device)
        wrapped_model = OnlyLogitsWrapper(self.model).to(self.device)
        wrapped_model.eval()

        try:
            with torch.no_grad():
                macs, thop_params = profile(
                    wrapped_model,
                    inputs=(dummy_input,),
                    verbose=False
                )
        except Exception as e:
            logging.exception("THOP profile failed when calculating aggregated global model complexity.")
            print("[ERROR] THOP profile failed:", str(e), flush=True)
            macs = 0
            thop_params = 0

        if was_training:
            self.model.train()

        param_stats = self._count_params(self.model)

        return {
            **param_stats,
            "thop_params": thop_params,
            "macs": macs,
            "flops": 2 * macs,
        }

    def log_global_model_complexity(self, round_idx, input_size=(1, 3, 32, 32), tag="AGGREGATED_GLOBAL_MODEL"):

        stats = self._calc_global_flops_params(input_size=input_size)

        msg = (
            f"\n========== {tag} Complexity ==========\n"
            f"Round: {round_idx + 1}\n"
            f"Stage: {self.args.stage}\n"
            f"Input size: {input_size}\n"
            f"Total params: {stats['total_params'] / 1e6:.6f} M\n"
            f"Weight params excluding arch params: {stats['weight_params'] / 1e6:.6f} M\n"
            f"Architecture params: {stats['arch_params'] / 1e6:.6f} M\n"
            f"Trainable params: {stats['trainable_params'] / 1e6:.6f} M\n"
            f"Params without auxiliary: {stats['params_no_auxiliary'] / 1e6:.6f} M\n"
            f"THOP params: {stats['thop_params'] / 1e6:.6f} M\n"
            f"MACs: {stats['macs'] / 1e6:.6f} M / {stats['macs'] / 1e9:.6f} G\n"
            f"FLOPs ~= 2 * MACs: {stats['flops'] / 1e6:.6f} M / {stats['flops'] / 1e9:.6f} G\n"
        )

        if self.args.stage == "search":
            msg += "NOTE: this is aggregated global search supernet complexity.\n"
        else:
            msg += "NOTE: this is aggregated global final model complexity.\n"

        msg += f"========== END {tag} Complexity ==========\n"

        print(msg, flush=True)
        logging.info(msg)

        return stats



    # ========= 收客户端模型 =========
    def add_local_trained_result(self, index, model_params, arch_params, sample_num, train_acc, train_loss):
        self.model_dict[index] = model_params
        self.arch_dict[index] = arch_params
        self.sample_num_dict[index] = sample_num

        self.train_acc_dict[index] = float(train_acc) if train_acc is not None else 0.0

        if torch.is_tensor(train_loss):
            train_loss = float(train_loss.detach().cpu().item())
        self.train_loss_dict[index] = float(train_loss) if train_loss is not None else 0.0

        self.flag_client_model_uploaded_dict[index] = True

    # ========= 收预测向量 =========
    def add_local_predicted_accs(self, index, predicted_accs):
        if predicted_accs is None:
            raise ValueError(f"Client {index} predicted_accs is None")
        if len(predicted_accs) != self.client_num:
            raise ValueError(f"Client {index} predicted_accs length {len(predicted_accs)} != client_num {self.client_num}")
        self.predicted_accs_dict[index] = predicted_accs
        self.flag_client_predicted_accs_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.client_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def check_whether_all_predicted_accs_receive(self):
        for idx in range(self.client_num):
            if not self.flag_client_predicted_accs_uploaded_dict[idx]:
                return False
        for idx in range(self.client_num):
            self.flag_client_predicted_accs_uploaded_dict[idx] = False
        return True

    def aggregate(self, round_idx):
        if round_idx < self.args.gp_train_round:
            averaged_weights = self.__aggregate_weight_fedavg()
            self.model.load_state_dict(averaged_weights)

            if self.args.stage == "search":
                averaged_alphas = self.__aggregate_alpha_fedavg()
                self.__update_arch(averaged_alphas)

                # Phase1 聚合完成后，统计全局模型 Params / FLOPs
                self.log_global_model_complexity(
                    round_idx=round_idx,
                    input_size=(1, 3, 32, 32),
                    tag="PHASE1_FEDAVG_GLOBAL_MODEL"
                )

                return averaged_weights, averaged_alphas

            # 非 search 阶段，Phase1 聚合完成后统计
            self.log_global_model_complexity(
                round_idx=round_idx,
                input_size=(1, 3, 32, 32),
                tag="PHASE1_FEDAVG_GLOBAL_MODEL"
            )

            return averaged_weights, None

        else:

            self.log_global_model_complexity(
                round_idx=round_idx,
                input_size=(1, 3, 32, 32),
                tag="PHASE2_PREDICTION_DRIVEN_GLOBAL_MODEL"
            )

            return result

    def _get_arch_param_ids(self, model):
        if hasattr(model, "arch_parameters"):
            try:
                return set(id(p) for p in model.arch_parameters())
            except Exception:
                return set()
        return set()

    def _count_params(self, model):
        arch_ids = self._get_arch_param_ids(model)

        total_params = 0
        weight_params = 0
        arch_params = 0
        trainable_params = 0
        params_no_auxiliary = 0

        for name, p in model.named_parameters():
            n = p.numel()

            total_params += n

            if p.requires_grad:
                trainable_params += n

            if id(p) in arch_ids:
                arch_params += n
            else:
                weight_params += n

            if "auxiliary" not in name:
                params_no_auxiliary += n

        return {
            "total_params": total_params,
            "weight_params": weight_params,
            "arch_params": arch_params,
            "trainable_params": trainable_params,
            "params_no_auxiliary": params_no_auxiliary,
        }

    def _calc_global_flops_params(self, input_size=(1, 3, 32, 32)):
        class OnlyLogitsWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                out = self.model(x)
                if isinstance(out, tuple):
                    return out[0]
                return out

        self.model.to(self.device)

        was_training = self.model.training
        self.model.eval()

        dummy_input = torch.randn(*input_size).to(self.device)
        wrapped_model = OnlyLogitsWrapper(self.model).to(self.device)
        wrapped_model.eval()

        try:
            with torch.no_grad():
                macs, thop_params = profile(
                    wrapped_model,
                    inputs=(dummy_input,),
                    verbose=False
                )
        except Exception as e:
            logging.exception("THOP profile failed when calculating aggregated global model complexity.")
            print("[ERROR] THOP profile failed:", str(e), flush=True)
            macs = 0
            thop_params = 0

        if was_training:
            self.model.train()

        param_stats = self._count_params(self.model)

        return {
            **param_stats,
            "thop_params": thop_params,
            "macs": macs,
            "flops": 2 * macs,
        }

    def log_global_model_complexity(self, round_idx, input_size=(1, 3, 32, 32), tag="AGGREGATED_GLOBAL_MODEL"):
        stats = self._calc_global_flops_params(input_size=input_size)

        msg = (
            f"\n========== {tag} Complexity ==========\n"
            f"Round: {round_idx + 1}\n"
            f"Stage: {self.args.stage}\n"
            f"Input size: {input_size}\n"
            f"Total params: {stats['total_params'] / 1e6:.6f} M\n"
            f"Weight params excluding arch params: {stats['weight_params'] / 1e6:.6f} M\n"
            f"Architecture params: {stats['arch_params'] / 1e6:.6f} M\n"
            f"Trainable params: {stats['trainable_params'] / 1e6:.6f} M\n"
            f"Params without auxiliary: {stats['params_no_auxiliary'] / 1e6:.6f} M\n"
            f"THOP params: {stats['thop_params'] / 1e6:.6f} M\n"
            f"MACs: {stats['macs'] / 1e6:.6f} M / {stats['macs'] / 1e9:.6f} G\n"
            f"FLOPs ~= 2 * MACs: {stats['flops'] / 1e6:.6f} M / {stats['flops'] / 1e9:.6f} G\n"
        )

        if self.args.stage == "search":
            msg += "NOTE: this is aggregated global search supernet complexity.\n"
        else:
            msg += "NOTE: this is aggregated global final model complexity.\n"

        msg += f"========== END {tag} Complexity ==========\n"

        print(msg, flush=True)
        logging.info(msg)

        return stats

    def __update_arch(self, alphas):
        for a_g, model_arch in zip(alphas, self.model.arch_parameters()):
            model_arch.data.copy_(a_g.data)

    # ========= Phase1: FedAvg =========
    def __aggregate_weight_fedavg(self):
        start_time = time.time()
        averaged_params = None
        for i in range(self.client_num):
            local_model_params = self.model_dict[i]
            w = self.sample_num_dict[i] / self.all_train_data_num

            if averaged_params is None:
                averaged_params = {k: v.detach().cpu() * w for k, v in local_model_params.items()}
            else:
                for k, v in local_model_params.items():
                    averaged_params[k] += v.detach().cpu() * w

        self.model_dict.clear()
        logging.info("FedAvg weight aggregation time: %.2fs", time.time() - start_time)
        return averaged_params

    def __aggregate_alpha_fedavg(self):
        start_time = time.time()
        averaged_alphas = None
        for i in range(self.client_num):
            local_alpha_params = self.arch_dict[i]
            w = self.sample_num_dict[i] / self.all_train_data_num

            if averaged_alphas is None:
                averaged_alphas = [p.detach().cpu() * w for p in local_alpha_params]
            else:
                for j, p in enumerate(local_alpha_params):
                    averaged_alphas[j] += p.detach().cpu() * w

        logging.info("FedAvg alpha aggregation time: %.2fs", time.time() - start_time)
        return averaged_alphas

    # ========= Phase2: Prediction-driven =========
    def __aggregate_prediction_driven(self):
        start_time = time.time()
        C = self.client_num

        # (1) data weight
        w_data = np.array([self.sample_num_dict[i] / self.all_train_data_num for i in range(C)], dtype=np.float64)

        # (2) pred matrix -> acc_bar
        pred_mat = np.zeros((C, C), dtype=np.float64)  # row=j, col=i
        for j in range(C):
            vec = self.predicted_accs_dict.get(j, None)
            if vec is None:
                raise RuntimeError(f"Missing predicted accs from client {j}")
            pred_mat[j, :] = np.array(vec, dtype=np.float64)

        acc_bar = pred_mat.mean(axis=0)
        acc_bar = np.nan_to_num(acc_bar, nan=0.0, posinf=0.0, neginf=0.0)
        acc_bar = np.maximum(acc_bar, 1e-12)
        w_pred = acc_bar / np.sum(acc_bar)

        # ★关键稳点：clip 防止极端权重（很常见能防掉点）
        w_pred = np.clip(w_pred, 1e-3, 1.0)
        w_pred = w_pred / np.sum(w_pred)

        zeta = float(getattr(self.args, "zeta", 0.4))
        eps = float(getattr(self.args, "epsilon", 0.6))
        w_final = zeta * w_data + eps * w_pred
        w_final = w_final / np.sum(w_final)

        logging.info("[Phase2] zeta=%.3f eps=%.3f", zeta, eps)
        logging.info("[Phase2] w_data=%s", w_data)
        logging.info("[Phase2] acc_bar=%s", acc_bar)
        logging.info("[Phase2] w_pred=%s", w_pred)
        logging.info("[Phase2] w_final=%s", w_final)

        # (3) aggregate weights
        averaged_weights = None
        for i in range(C):
            local_model_params = self.model_dict[i]
            w = float(w_final[i])
            if averaged_weights is None:
                averaged_weights = {k: v.detach().cpu() * w for k, v in local_model_params.items()}
            else:
                for k, v in local_model_params.items():
                    averaged_weights[k] += v.detach().cpu() * w

        self.model.load_state_dict(averaged_weights)

        averaged_alphas = None
        if self.args.stage == "search":
            for i in range(C):
                local_alpha_params = self.arch_dict[i]
                w = float(w_final[i])
                if averaged_alphas is None:
                    averaged_alphas = [p.detach().cpu() * w for p in local_alpha_params]
                else:
                    for j, p in enumerate(local_alpha_params):
                        averaged_alphas[j] += p.detach().cpu() * w
            self.__update_arch(averaged_alphas)

        # cleanup
        self.model_dict.clear()
        self.arch_dict.clear()
        self.predicted_accs_dict.clear()

        logging.info("Prediction-driven aggregation time: %.2fs", time.time() - start_time)
        return averaged_weights, averaged_alphas

    # ========= server val =========
    def infer(self, round_idx):
        if round_idx % self.args.frequency_of_the_test != 0 and round_idx != self.args.comm_round - 1:
            return

        self.model.eval()
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss().to(self.device)

        test_correct, test_loss, test_n = 0.0, 0.0, 0.0
        with torch.no_grad():
            for x, target in self.test_global:
                x = x.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                pred = self.model(x)
                if self.args.stage == "train":
                    loss = criterion(pred[0], target)
                    _, predicted = torch.max(pred[0], 1)
                else:
                    loss = criterion(pred, target)
                    _, predicted = torch.max(pred, 1)

                test_correct += predicted.eq(target).sum().item()
                test_loss += loss.item() * target.size(0)
                test_n += target.size(0)

        self.test_acc_avg = test_correct / max(test_n, 1.0)
        self.test_loss_avg = test_loss / max(test_n, 1.0)

        if self.test_acc_avg > self.best_accuracy:
            self.best_accuracy = self.test_acc_avg

    def statistics(self, round_idx):
        accs = list(self.train_acc_dict.values())
        losses = list(self.train_loss_dict.values())
        self.train_acc_avg = float(np.mean(accs)) if accs else 0.0
        self.train_loss_avg = float(np.mean(losses)) if losses else 0.0

        logging.info("Round %3d | client_train_acc_avg=%.6f client_train_loss_avg=%.6f | "
                     "server_val_acc=%.6f server_val_loss=%.6f | best=%.6f",
                     round_idx + 1,
                     self.train_acc_avg, self.train_loss_avg,
                     self.test_acc_avg, self.test_loss_avg,
                     self.best_accuracy)

    def record_model_global_architecture(self, round_idx):
        if self.args.stage != "search":
            return
        genotype, normal_cnn_count, reduce_cnn_count = self.model.genotype()
        logging.info("(n:%d,r:%d) genotype=%s", normal_cnn_count, reduce_cnn_count, genotype)
