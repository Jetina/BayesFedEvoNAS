# -*- coding: utf-8 -*-
import logging
import random
from itertools import cycle

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
import gpytorch
import torch
import torch.nn.functional as F
from torch import nn
from thop import profile

from darts import utils, genotypes
from darts.architect import Architect
from darts.model import NetworkCIFAR
from darts.model_search import Network


class SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class FedNASTrainer(object):
    def __init__(self, client_index, train_local, test_local, local_sample_number, all_train_data_num, device, args):
        self.client_index = client_index
        self.train_local = train_local
        self.test_local = test_local
        self.local_sample_number = local_sample_number
        self.all_train_data_num = all_train_data_num
        self.device = device
        self.args = args

        self.num_classes = 10
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.model = self.init_model().to(self.device)
        # # ====== 强制打印参数量和 FLOPs，search 阶段也打印 ======
        # print("\n[DEBUG] Start calculating Params and FLOPs...", flush=True)
        # self.log_model_complexity(input_size=(1, 3, 32, 32))
        # print("[DEBUG] Finish calculating Params and FLOPs.\n", flush=True)

        self.scaler = GradScaler(enabled=getattr(self.args, "amp", False) and torch.cuda.is_available())
        torch.backends.cudnn.benchmark = True

        self.round_idx = 0
        self.best_population = []  # list of (val_acc, alphas_cpu_list)

        self._init_gpytorch_model()

    def set_round_idx(self, round_idx):
        self.round_idx = round_idx

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

    def _get_arch_param_ids(self, model):
        """
        返回 architecture parameters 的 id 集合。
        search 阶段 model 有 arch_parameters()。
        final NetworkCIFAR 通常没有 arch_parameters()。
        """
        if hasattr(model, "arch_parameters"):
            try:
                return set(id(p) for p in model.arch_parameters())
            except Exception:
                return set()
        return set()

    def _count_params(self, model):
        """
        统计参数量：
        1. total_params: 所有参数，包括 architecture parameters
        2. weight_params: 普通网络权重参数，不包括 architecture parameters
        3. arch_params: architecture parameters
        4. trainable_params: requires_grad=True 的参数
        5. params_no_auxiliary: 排除 auxiliary_head 的参数
        """
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

    def _calc_flops_params(self, input_size=(1, 3, 32, 32)):
        """
        统计当前 self.model 的 Params 和 FLOPs。

        search 阶段：
            统计的是 supernet 的复杂度。
            如果 MixedOp forward 会执行所有候选操作，那么 FLOPs 会包含所有候选操作的计算量。

        final 阶段：
            统计的是 NetworkCIFAR 的最终离散结构复杂度。

        注意：
            thop 返回的是 MACs。
            严格 FLOPs 通常约等于 2 * MACs。
            很多 NAS 论文写 FLOPs，但实际报的是 MACs。
        """

        class OnlyLogitsWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                out = self.model(x)
                if isinstance(out, tuple):
                    return out[0]
                return out

        was_training = self.model.training
        self.model.eval()

        dummy_input = torch.randn(*input_size).to(self.device)

        wrapped_model = OnlyLogitsWrapper(self.model).to(self.device)
        wrapped_model.eval()

        with torch.no_grad():
            macs, thop_params = profile(
                wrapped_model,
                inputs=(dummy_input,),
                verbose=False
            )

        if was_training:
            self.model.train()

        param_stats = self._count_params(self.model)

        return {
            **param_stats,
            "thop_params": thop_params,
            "macs": macs,
            "flops": 2 * macs,
        }

    def log_model_complexity(self, input_size=(1, 3, 32, 32)):
        stats = self._calc_flops_params(input_size=input_size)

        logging.info("========== Model Complexity ==========")
        logging.info("Client %d, stage=%s", self.client_index, self.args.stage)
        logging.info("Input size: %s", str(input_size))

        logging.info("Total params: %.6f M", stats["total_params"] / 1e6)
        logging.info("Weight params excluding arch params: %.6f M", stats["weight_params"] / 1e6)
        logging.info("Architecture params: %.6f M", stats["arch_params"] / 1e6)
        logging.info("Trainable params: %.6f M", stats["trainable_params"] / 1e6)
        logging.info("Params without auxiliary: %.6f M", stats["params_no_auxiliary"] / 1e6)
        logging.info("THOP params: %.6f M", stats["thop_params"] / 1e6)

        logging.info("MACs: %.6f M / %.6f G", stats["macs"] / 1e6, stats["macs"] / 1e9)
        logging.info("FLOPs ~= 2 * MACs: %.6f M / %.6f G", stats["flops"] / 1e6, stats["flops"] / 1e9)

        if self.args.stage == "search":
            logging.info(
                "NOTE: search stage complexity is supernet complexity, not final discrete architecture complexity."
            )

        logging.info("======================================")

    def update_model(self, weights):
        self.model.load_state_dict(weights)

    def update_arch(self, alphas):
        for a_g, model_arch in zip(alphas, self.model.arch_parameters()):
            model_arch.data.copy_(a_g.data)

    # ========= arch 向量化 =========
    def _arch_to_vector(self, alphas):
        return torch.cat([a.detach().view(-1) for a in alphas]).to(self.device)

    def _vector_to_arch(self, vector):
        arch_params = []
        ptr = 0
        template = self.model.arch_parameters()
        for alpha_t in template:
            shape = alpha_t.shape
            size = int(np.prod(shape))
            param_data = vector[ptr: ptr + size].view(shape)
            arch_params.append(param_data.clone().detach().to(self.device))
            ptr += size
        return arch_params

    # ========= GP =========
    def _init_gpytorch_model(self):
        with torch.no_grad():
            arch_dim = sum(p.numel() for p in self.model.arch_parameters())

        inducing_num = int(getattr(self.args, "gp_inducing_points", 16))
        inducing_points = torch.randn(inducing_num, arch_dim, device=self.device)

        self.gp_model = SparseGPModel(inducing_points=inducing_points).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.gp_optimizer = torch.optim.Adam(
            [{'params': self.gp_model.parameters()}, {'params': self.likelihood.parameters()}],
            lr=float(getattr(self.args, "gp_lr", 0.01))
        )

    def _train_gpytorch_model(self, train_x, train_y, steps=1):
        self.gp_model.train()
        self.likelihood.train()
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.gp_model, num_data=len(train_x))

        for _ in range(steps):
            self.gp_optimizer.zero_grad()
            output = self.gp_model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            self.gp_optimizer.step()

    def predict_architectures(self, arch_collection, is_vector_input=False):
        self.gp_model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            if not is_vector_input:
                X = torch.stack([self._arch_to_vector(arch) for arch in arch_collection]).to(self.device)
            else:
                X = arch_collection.to(self.device)
            pred = self.likelihood(self.gp_model(X))
            return pred.mean.detach().cpu().tolist()

    # ========= search 入口 =========
    def search(self):
        self.model.to(self.device)
        self.model.train()

        if self.round_idx < self.args.gp_train_round:
            return self._search_phase1()
        return self._search_phase2_evolutionary()

    # ========= 搜索优化器（返回 weight_params 供 clip） =========
    def _get_search_optimizers(self):
        arch_parameters = self.model.arch_parameters()
        arch_ids = list(map(id, arch_parameters))

        weight_params = list(filter(lambda p: id(p) not in arch_ids, self.model.parameters()))

        optimizer = torch.optim.SGD(
            weight_params,
            self.args.learning_rate,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )
        architect = Architect(self.model, self.criterion, self.args, self.device)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(self.args.epochs), eta_min=self.args.learning_rate_min
        )
        return optimizer, scheduler, architect, weight_params

    # ========= Phase1 =========
    def _search_phase1(self):
        optimizer, scheduler, architect, weight_params = self._get_search_optimizers()

        GP_WARMUP = int(getattr(self.args, "gp_warmup_samples", 20))
        GP_EVERY = int(getattr(self.args, "gp_update_every", 2))
        GP_STEPS = int(getattr(self.args, "gp_train_steps", 1))
        TOPK = int(getattr(self.args, "elite_pool_size", 50))

        local_epoch_val_accs = []
        local_epoch_losses = []

        for epoch in range(self.args.epochs):
            train_acc, train_loss = self._local_search_one_epoch(
                self.train_local, self.test_local, self.model, architect,
                self.criterion, optimizer, epoch, weight_params
            )
            scheduler.step()

            val_acc, _ = self._local_infer(self.test_local, max_batches=getattr(self.args, "gp_eval_batches", None))
            local_epoch_val_accs.append(val_acc)
            local_epoch_losses.append(train_loss)

            with torch.no_grad():
                alphas_snapshot = [a.detach().cpu().clone() for a in self.model.arch_parameters()]
                self.best_population.append((val_acc, alphas_snapshot))

            logging.info("Client %d R%d E%d: train_acc=%.4f val_acc=%.4f",
                         self.client_index, self.round_idx + 1, epoch + 1, train_acc, val_acc)

        self.best_population.sort(key=lambda x: x[0], reverse=True)
        self.best_population = self.best_population[:TOPK]

        warmed_up = len(self.best_population) >= GP_WARMUP
        update_now = ((self.round_idx + 1) % GP_EVERY == 0)

        if warmed_up and update_now:
            xs = torch.stack([self._arch_to_vector([p.to(self.device) for p in alpha]) for _, alpha in self.best_population])
            ys = torch.tensor([acc for acc, _ in self.best_population], device=self.device, dtype=torch.float32)
            self._train_gpytorch_model(xs, ys, steps=GP_STEPS)

        weights = self.model.cpu().state_dict()
        alphas = [a.detach().cpu() for a in self.model.arch_parameters()]

        final_val_acc = float(local_epoch_val_accs[-1]) if local_epoch_val_accs else 0.0
        avg_loss = float(np.mean(local_epoch_losses)) if local_epoch_losses else 0.0

        return weights, alphas, self.local_sample_number, final_val_acc, avg_loss

    # ========= Phase2: EA =========
    def _search_phase2_evolutionary(self):
        optimizer, scheduler, architect, weight_params = self._get_search_optimizers()

        TOPK = int(getattr(self.args, "elite_pool_size", 50))
        G = int(getattr(self.args, "evo_generations", 30))
        tournament_k = int(getattr(self.args, "evo_tournament_k", 10))

        round_new = []
        for epoch in range(self.args.epochs):
            _, _ = self._local_search_one_epoch(
                self.train_local, self.test_local, self.model, architect,
                self.criterion, optimizer, epoch, weight_params
            )
            scheduler.step()

            val_acc, _ = self._local_infer(self.test_local, max_batches=getattr(self.args, "gp_eval_batches", None))
            with torch.no_grad():
                alphas_snapshot = [a.detach().cpu().clone() for a in self.model.arch_parameters()]
                round_new.append((val_acc, alphas_snapshot))

        self.best_population.extend(round_new)
        self.best_population.sort(key=lambda x: x[0], reverse=True)
        self.best_population = self.best_population[:TOPK]

        pop_alphas = [alpha for _, alpha in self.best_population]
        with torch.no_grad():
            pop_vec = torch.stack([self._arch_to_vector([p.to(self.device) for p in alpha]) for alpha in pop_alphas])
            pop_pred = self.predict_architectures(pop_vec, is_vector_input=True)
        population = list(zip(pop_pred, pop_alphas))

        for gen in range(G):
            parents = [self._tournament_selection(population, k=tournament_k) for _ in range(len(population))]
            offspring_vecs = []

            for i in range(0, len(parents), 2):
                if i + 1 >= len(parents):
                    break
                p1 = self._arch_to_vector([p.to(self.device) for p in parents[i][1]])
                p2 = self._arch_to_vector([p.to(self.device) for p in parents[i + 1][1]])
                c1, c2 = self._crossover(p1, p2, rate=float(getattr(self.args, "evo_crossover_rate", 0.9)))
                c1 = self._mutate(c1, rate=float(getattr(self.args, "evo_mutate_rate", 0.1)),
                                  sigma=float(getattr(self.args, "evo_mutate_sigma", 0.1)))
                c2 = self._mutate(c2, rate=float(getattr(self.args, "evo_mutate_rate", 0.1)),
                                  sigma=float(getattr(self.args, "evo_mutate_sigma", 0.1)))
                offspring_vecs.extend([c1, c2])

            if not offspring_vecs:
                continue

            with torch.no_grad():
                off_X = torch.stack(offspring_vecs).to(self.device)
                off_pred = self.predict_architectures(off_X, is_vector_input=True)
                offspring = list(zip(off_pred, [self._vector_to_arch(v) for v in off_X]))

            combined = population + offspring
            combined.sort(key=lambda x: x[0], reverse=True)
            population = combined[:len(population)]

        best_pred_acc, best_arch = population[0]
        self.update_arch([p.to(self.device) for p in best_arch])

        # 进化后再训练一轮 weights
        post_acc, post_loss = self._local_train_one_epoch(self.train_local, optimizer, weight_params)

        weights = self.model.cpu().state_dict()
        alphas = [a.detach().cpu() for a in self.model.arch_parameters()]
        return weights, alphas, self.local_sample_number, float(post_acc), float(post_loss)

    # ========= 1 epoch search =========
    # def _local_search_one_epoch(self, train_queue, valid_queue, model, architect, criterion, optimizer, epoch, weight_params):
    #     model.train()
    #     objs, top1 = utils.AvgrageMeter(), utils.AvgrageMeter()
    #     valid_iter = cycle(valid_queue)
    #
    #     arch_interval = max(1, int(getattr(self.args, "arch_step_interval", 2)))
    #
    #     for step, (inp, tgt) in enumerate(train_queue):
    #         inp = inp.to(self.device, non_blocking=True)
    #         tgt = tgt.to(self.device, non_blocking=True)
    #
    #         if step % arch_interval == 0:
    #             in_s, tg_s = next(valid_iter)
    #             in_s = in_s.to(self.device, non_blocking=True)
    #             tg_s = tg_s.to(self.device, non_blocking=True)
    #             architect.step_v2(
    #                 inp, tgt, in_s, tg_s,
    #                 self.args.lambda_train_regularizer,
    #                 self.args.lambda_valid_regularizer
    #             )
    #
    #         optimizer.zero_grad(set_to_none=True)
    #         with autocast(enabled=self.scaler.is_enabled()):
    #             logits = model(inp)
    #             loss = criterion(logits, tgt)
    #
    #         self.scaler.scale(loss).backward()
    #
    #         # ★关键修复：只裁剪 weight_params，不裁剪 arch params
    #         self.scaler.unscale_(optimizer)
    #         nn.utils.clip_grad_norm_(weight_params, self.args.grad_clip)
    #
    #         self.scaler.step(optimizer)
    #         self.scaler.update()
    #
    #         prec1, _ = utils.accuracy(logits, tgt, topk=(1, min(5, self.num_classes)))
    #         objs.update(loss.item(), inp.size(0))
    #         top1.update(prec1.item(), inp.size(0))
    #
    #         if step % self.args.report_freq == 0:
    #             logging.info("client=%d epoch=%d step=%d loss=%.4f top1=%.4f",
    #                          self.client_index, epoch, step, objs.avg, top1.avg)
    #
    #     return float(top1.avg / 100.0), float(objs.avg)
    def _local_search_one_epoch(self, train_queue, valid_queue, model, architect, criterion, optimizer, epoch,
                                weight_params):
        model.train()
        objs, top1 = utils.AvgrageMeter(), utils.AvgrageMeter()
        valid_iter = cycle(valid_queue)

        arch_interval = max(1, int(getattr(self.args, "arch_step_interval", 2)))
        grad_accum_steps = max(1, int(getattr(self.args, "grad_accum_steps", 1)))

        optimizer.zero_grad(set_to_none=True)

        for step, (inp, tgt) in enumerate(train_queue):
            inp = inp.to(self.device, non_blocking=True)
            tgt = tgt.to(self.device, non_blocking=True)

            # ====== 架构更新：仍按 micro-step 频率，不改变你的方法逻辑 ======
            if step % arch_interval == 0:
                in_s, tg_s = next(valid_iter)
                in_s = in_s.to(self.device, non_blocking=True)
                tg_s = tg_s.to(self.device, non_blocking=True)
                architect.step_v2(
                    inp, tgt, in_s, tg_s,
                    self.args.lambda_train_regularizer,
                    self.args.lambda_valid_regularizer
                )

            # ====== 梯度累计：loss 需要除以 grad_accum_steps ======
            with autocast(enabled=self.scaler.is_enabled()):
                logits = model(inp)
                loss = criterion(logits, tgt)
                loss_to_backward = loss / float(grad_accum_steps)

            self.scaler.scale(loss_to_backward).backward()

            # 是否该做一次 optimizer step
            do_step = ((step + 1) % grad_accum_steps == 0)

            if do_step:
                # 只在 step 时 unscale + clip + step
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(weight_params, self.args.grad_clip)

                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # ====== 统计：用原始 loss（没除过的）更直观 ======
            prec1, _ = utils.accuracy(logits, tgt, topk=(1, min(5, self.num_classes)))
            objs.update(float(loss.detach().item()), inp.size(0))
            top1.update(float(prec1.detach().item()), inp.size(0))

            if step % self.args.report_freq == 0:
                logging.info(
                    "client=%d epoch=%d step=%d loss=%.4f top1=%.4f (grad_accum=%d)",
                    self.client_index, epoch, step, objs.avg, top1.avg, grad_accum_steps
                )

        return float(top1.avg / 100.0), float(objs.avg)

    def _local_train_one_epoch(self, train_queue, optimizer, weight_params):
        self.model.train()
        objs, top1 = utils.AvgrageMeter(), utils.AvgrageMeter()

        grad_accum_steps = max(1, int(getattr(self.args, "grad_accum_steps", 1)))
        optimizer.zero_grad(set_to_none=True)

        for step, (inp, tgt) in enumerate(train_queue):
            inp = inp.to(self.device, non_blocking=True)
            tgt = tgt.to(self.device, non_blocking=True)

            with autocast(enabled=self.scaler.is_enabled()):
                out = self.model(inp)
                if isinstance(out, tuple):
                    logits, logits_aux = out
                else:
                    logits, logits_aux = out, None

                loss = self.criterion(logits, tgt)
                if getattr(self.args, "auxiliary", False) and logits_aux is not None:
                    loss = loss + self.args.auxiliary_weight * self.criterion(logits_aux, tgt)

                loss_to_backward = loss / float(grad_accum_steps)

            self.scaler.scale(loss_to_backward).backward()

            do_step = ((step + 1) % grad_accum_steps == 0)

            if do_step:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(weight_params, self.args.grad_clip)

                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)

            prec1, _ = utils.accuracy(logits, tgt, topk=(1, min(5, self.num_classes)))
            objs.update(float(loss.detach().item()), inp.size(0))
            top1.update(float(prec1.detach().item()), inp.size(0))

        return float(top1.avg / 100.0), float(objs.avg)

    def _local_infer(self, valid_queue, max_batches=None):
        self.model.eval()
        correct, total, loss_sum, batches = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for inp, tgt in valid_queue:
                inp = inp.to(self.device, non_blocking=True)
                tgt = tgt.to(self.device, non_blocking=True)

                logits = self.model(inp)
                if isinstance(logits, tuple):
                    logits = logits[0]

                loss = self.criterion(logits, tgt)
                _, pred = torch.max(logits, 1)

                correct += pred.eq(tgt).sum().item()
                total += tgt.size(0)
                loss_sum += loss.item() * tgt.size(0)

                batches += 1
                if max_batches is not None and batches >= int(max_batches):
                    break

        acc = correct / max(total, 1.0)
        avg_loss = loss_sum / max(total, 1.0)
        return float(acc), float(avg_loss)

    # ========= EA tools =========
    def _tournament_selection(self, population, k=10):
        best = None
        for _ in range(k):
            ind = random.choice(population)
            if best is None or ind[0] > best[0]:
                best = ind
        return best

    def _crossover(self, p1_vec, p2_vec, rate=0.9):
        if random.random() < rate and len(p1_vec) > 2:
            point = random.randint(1, len(p1_vec) - 2)
            c1 = torch.cat([p1_vec[:point], p2_vec[point:]])
            c2 = torch.cat([p2_vec[:point], p1_vec[point:]])
            return c1, c2
        return p1_vec.clone(), p2_vec.clone()

    def _mutate(self, ind_vec, rate=0.1, mu=0.0, sigma=0.1):
        if random.random() < rate:
            noise = torch.randn(ind_vec.shape, device=self.device) * sigma + mu
            return ind_vec + noise
        return ind_vec
