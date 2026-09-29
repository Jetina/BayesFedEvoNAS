# -*- coding: utf-8 -*-
import logging
import sys
import time
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from communication.com_manager import CommunicationManager
from communication.mpi_message import MPIMessage
from communication.observer import Observer

MSG_TYPE_S2C_ARCH_COLLECTION_TO_CLIENT = 4
MSG_TYPE_C2S_SEND_PREDICTED_ACCS_TO_SERVER = 5


class ServerMananger(Observer):
    def __init__(self, args, comm, rank, size, round_num, aggregator):
        self.args = args
        self.size = size
        self.rank = rank
        self.round_num = round_num
        self.round_idx = 0

        self.com_manager = CommunicationManager(comm, rank, size, node_type="server")
        self.com_manager.add_observer(self)
        self.aggregator = aggregator

        self.round_start_time = time.time()
        self.total_start_time = time.time()

        # 输出目录：result/<dataset>_C<client_number>/
        dataset = getattr(self.args, "dataset", "exp")
        client_num = int(getattr(self.args, "client_number", (self.size - 1)))
        root = getattr(self.args, "result_root", "result")

        self.exp_dir = os.path.join(root, f"{dataset}_C{client_num}")
        os.makedirs(self.exp_dir, exist_ok=True)

        self.static_path = os.path.join(self.exp_dir, "static.txt")
        self.curve_path = os.path.join(self.exp_dir, "val_acc_curve.png")

        self.round_hist = []
        self.val_acc_hist = []

        if not os.path.exists(self.static_path):
            with open(self.static_path, "w", encoding="utf-8") as f:
                f.write("# round | phase | client_train_acc_avg | client_train_loss_avg | "
                        "server_val_acc | server_val_loss | best_acc\n")

        logging.info("[RESULT] experiment dir = %s", self.exp_dir)

    def receive_message(self, msg_type, msg_params) -> None:
        if msg_type == MPIMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER:
            self.__handle_model_from_client(msg_params)
        elif msg_type == MSG_TYPE_C2S_SEND_PREDICTED_ACCS_TO_SERVER:
            self.__handle_predicted_accs(msg_params)

    def run(self):
        self.__broadcast_initial_config_to_client()
        self.com_manager.handle_receive_message()

    def __broadcast_initial_config_to_client(self):
        global_model = self.aggregator.get_model()
        global_model_params = global_model.state_dict()
        global_arch_params = []
        if self.args.stage == "search":
            global_arch_params = global_model.arch_parameters()

        msg = MPIMessage()
        msg.add(MPIMessage.MSG_ARG_KEY_OPERATION, MPIMessage.MSG_OPERATION_BROADCAST)
        msg.add(MPIMessage.MSG_ARG_KEY_TYPE, MPIMessage.MSG_TYPE_S2C_INIT_CONFIG)
        msg.add(MPIMessage.MSG_ARG_KEY_SENDER, 0)
        msg.add(MPIMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        msg.add(MPIMessage.MSG_ARG_KEY_ARCH_PARAMS, global_arch_params)
        self.com_manager.send_broadcast_collective_message(msg)

    def _append_static(self, phase: str):
        r = self.round_idx + 1
        c_train_acc_avg = float(getattr(self.aggregator, "train_acc_avg", 0.0))
        c_train_loss_avg = float(getattr(self.aggregator, "train_loss_avg", 0.0))
        server_val_acc = float(getattr(self.aggregator, "test_acc_avg", 0.0))
        server_val_loss = float(getattr(self.aggregator, "test_loss_avg", 0.0))
        best_acc = float(getattr(self.aggregator, "best_accuracy", 0.0))

        with open(self.static_path, "a", encoding="utf-8") as f:
            f.write(
                f"{r} | {phase} | "
                f"{c_train_acc_avg:.6f} | {c_train_loss_avg:.6f} | "
                f"{server_val_acc:.6f} | {server_val_loss:.6f} | {best_acc:.6f}\n"
            )

        self.round_hist.append(r)
        self.val_acc_hist.append(server_val_acc)

    def _plot_curve(self):
        if not self.round_hist:
            return
        plt.figure()
        plt.plot(self.round_hist, self.val_acc_hist, marker="o")
        plt.xlabel("Round")
        plt.ylabel("Server Validation Accuracy")
        plt.title("Server Validation Accuracy vs Round")
        plt.grid(True)
        plt.savefig(self.curve_path, dpi=200, bbox_inches="tight")
        plt.close()
        logging.info("[PLOT] saved to %s", self.curve_path)

    def __handle_model_from_client(self, msg_params):
        pid = msg_params.get(MPIMessage.MSG_ARG_KEY_SENDER)

        model_params = msg_params.get(MPIMessage.MSG_ARG_KEY_MODEL_PARAMS)
        arch_params = msg_params.get(MPIMessage.MSG_ARG_KEY_ARCH_PARAMS)
        local_n = msg_params.get(MPIMessage.MSG_ARG_KEY_NUM_SAMPLES)

        train_acc = msg_params.get(MPIMessage.MSG_ARG_KEY_LOCAL_TRAINING_ACC)
        train_loss = msg_params.get(MPIMessage.MSG_ARG_KEY_LOCAL_TRAINING_LOSS)

        # ★关键修复：只传 6 个参数
        self.aggregator.add_local_trained_result(
            pid - 1, model_params, arch_params, local_n, train_acc, train_loss
        )

        if not self.aggregator.check_whether_all_receive():
            return

        # Phase1：直接聚合 + 广播
        if self.round_idx < self.args.gp_train_round:
            global_model_params, global_arch_params = self.aggregator.aggregate(self.round_idx)
            self.aggregator.infer(self.round_idx)
            self.aggregator.statistics(self.round_idx)

            if self.args.stage == "search":
                self.aggregator.record_model_global_architecture(self.round_idx)

            self._append_static("phase1")
            self._log_round_time()
            self.__broadcast_model_to_all_clients(global_model_params, global_arch_params)

            self.round_idx += 1
            if self.round_idx == self.round_num:
                self.__finish()
            return

        # Phase2：广播架构集合（按 client id 顺序）
        arch_collection = [self.aggregator.arch_dict[i] for i in range(self.aggregator.client_num)]
        self.__broadcast_arch_collection_to_clients(arch_collection)

    def __handle_predicted_accs(self, msg_params):
        pid = msg_params.get(MPIMessage.MSG_ARG_KEY_SENDER)
        predicted_accs = msg_params.get("predicted_accs")

        self.aggregator.add_local_predicted_accs(pid - 1, predicted_accs)

        if not self.aggregator.check_whether_all_predicted_accs_receive():
            return

        global_model_params, global_arch_params = self.aggregator.aggregate(self.round_idx)
        self.aggregator.infer(self.round_idx)
        self.aggregator.statistics(self.round_idx)

        if self.args.stage == "search":
            self.aggregator.record_model_global_architecture(self.round_idx)

        self._append_static("phase2")
        self._log_round_time()
        self.__broadcast_model_to_all_clients(global_model_params, global_arch_params)

        self.round_idx += 1
        if self.round_idx == self.round_num:
            self.__finish()

    def __broadcast_arch_collection_to_clients(self, arch_collection):
        msg = MPIMessage()
        msg.add(MPIMessage.MSG_ARG_KEY_OPERATION, MPIMessage.MSG_OPERATION_BROADCAST)
        msg.add(MPIMessage.MSG_ARG_KEY_TYPE, MSG_TYPE_S2C_ARCH_COLLECTION_TO_CLIENT)
        msg.add(MPIMessage.MSG_ARG_KEY_SENDER, 0)
        msg.add(MPIMessage.MSG_ARG_KEY_ARCH_PARAMS, arch_collection)
        self.com_manager.send_broadcast_collective_message(msg)

    def __broadcast_model_to_all_clients(self, global_model_params, global_arch_params):
        msg = MPIMessage()
        msg.add(MPIMessage.MSG_ARG_KEY_OPERATION, MPIMessage.MSG_OPERATION_BROADCAST)
        msg.add(MPIMessage.MSG_ARG_KEY_TYPE, MPIMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT)
        msg.add(MPIMessage.MSG_ARG_KEY_SENDER, 0)
        msg.add(MPIMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        msg.add(MPIMessage.MSG_ARG_KEY_ARCH_PARAMS, global_arch_params)
        self.com_manager.send_broadcast_collective_message(msg)

    def _log_round_time(self):
        t = time.time()
        per_round = t - self.round_start_time
        total = t - self.total_start_time
        logging.info("---- Round %d finished | time=%.2fs | total=%.2fh ----",
                     self.round_idx + 1, per_round, total / 3600.0)
        self.round_start_time = time.time()

    def __finish(self):
        try:
            self._plot_curve()
        except Exception as e:
            logging.exception("plot failed: %s", e)

        total = time.time() - self.total_start_time
        logging.info("ALL DONE. total=%.2fh result_dir=%s", total / 3600.0, self.exp_dir)

        self.com_manager.stop_receive_message()
        sys.exit()
