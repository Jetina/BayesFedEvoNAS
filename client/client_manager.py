# -*- coding: utf-8 -*-
import logging
import sys
import time

from communication.com_manager import CommunicationManager
from communication.mpi_message import MPIMessage
from communication.observer import Observer

MSG_TYPE_S2C_ARCH_COLLECTION_TO_CLIENT = 4
MSG_TYPE_C2S_SEND_PREDICTED_ACCS_TO_SERVER = 5


class ClientMananger(Observer):
    def __init__(self, args, comm, rank, size, round_num, trainer):
        self.args = args
        self.size = size
        self.rank = rank
        self.com_manager = CommunicationManager(comm, rank, size, node_type="client")
        self.com_manager.add_observer(self)
        self.trainer = trainer
        self.num_rounds = round_num
        self.round_idx = 0

    def receive_message(self, msg_type, msg_params) -> None:
        logging.info("receive_message. rank_id=%d, msg_type=%s", self.rank, str(msg_type))

        if msg_type == MPIMessage.MSG_TYPE_S2C_INIT_CONFIG:
            self.__handle_msg_client_receive_config(msg_params)

        elif msg_type == MPIMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT:
            self.__handle_msg_client_receive_model_from_server(msg_params)

        elif msg_type == MSG_TYPE_S2C_ARCH_COLLECTION_TO_CLIENT:
            self.__handle_msg_receive_arch_collection(msg_params)

    def run(self):
        self.com_manager.handle_receive_message()

    def __handle_msg_client_receive_config(self, msg_params):
        global_model_params = msg_params.get(MPIMessage.MSG_ARG_KEY_MODEL_PARAMS)
        arch_params = msg_params.get(MPIMessage.MSG_ARG_KEY_ARCH_PARAMS)

        self.trainer.update_model(global_model_params)
        if self.args.stage == "search" and arch_params:
            self.trainer.update_arch(arch_params)

        self.round_idx = 0
        self.trainer.set_round_idx(self.round_idx)
        self.__train()

    def __handle_msg_client_receive_model_from_server(self, msg_params):
        model_params = msg_params.get(MPIMessage.MSG_ARG_KEY_MODEL_PARAMS)
        arch_params = msg_params.get(MPIMessage.MSG_ARG_KEY_ARCH_PARAMS)

        self.trainer.update_model(model_params)
        if self.args.stage == "search" and arch_params:
            self.trainer.update_arch(arch_params)

        self.round_idx += 1
        self.trainer.set_round_idx(self.round_idx)

        self.__train()

        if self.round_idx == self.num_rounds - 1:
            self.__finish()

    def __handle_msg_receive_arch_collection(self, msg_params):
        """
        Phase2: server 发来 arch_collection（按 client id 顺序）
        client 返回 predicted_accs（同顺序）
        """
        logging.info("Client %d: Received arch collection from server.", self.rank)

        arch_collection = msg_params.get(MPIMessage.MSG_ARG_KEY_ARCH_PARAMS)
        predicted_accs = self.trainer.predict_architectures(arch_collection)

        msg = MPIMessage()
        msg.add(MPIMessage.MSG_ARG_KEY_OPERATION, MPIMessage.MSG_OPERATION_SEND)
        msg.add(MPIMessage.MSG_ARG_KEY_TYPE, MSG_TYPE_C2S_SEND_PREDICTED_ACCS_TO_SERVER)
        msg.add(MPIMessage.MSG_ARG_KEY_SENDER, self.rank)
        msg.add(MPIMessage.MSG_ARG_KEY_RECEIVER, 0)
        msg.add("predicted_accs", predicted_accs)

        self.com_manager.send_message(msg)
        logging.info("Client %d: Sent predicted_accs back to server.", self.rank)

    def __train(self):
        logging.info("#######training########### round_id=%d", self.round_idx)
        start_time = time.time()

        if self.args.stage == "search":
            weights, alphas, local_sample_num, train_acc, train_loss = self.trainer.search()
        else:
            weights, local_sample_num, train_acc, train_loss = self.trainer.train()
            alphas = []

        train_finished_time = time.time()
        logging.info("local training time cost: %.2fs", (train_finished_time - start_time))

        self.__send_msg_fedavg_send_model_to_server(weights, alphas, local_sample_num, train_acc, train_loss)

        communication_finished_time = time.time()
        logging.info("local communication time cost: %.2fs", (communication_finished_time - train_finished_time))

    def __send_msg_fedavg_send_model_to_server(self, weights, alphas, local_sample_num, valid_acc, valid_loss):
        msg = MPIMessage()
        msg.add(MPIMessage.MSG_ARG_KEY_OPERATION, MPIMessage.MSG_OPERATION_SEND)
        msg.add(MPIMessage.MSG_ARG_KEY_TYPE, MPIMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER)
        msg.add(MPIMessage.MSG_ARG_KEY_SENDER, self.rank)
        msg.add(MPIMessage.MSG_ARG_KEY_RECEIVER, 0)

        msg.add(MPIMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        msg.add(MPIMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        msg.add(MPIMessage.MSG_ARG_KEY_ARCH_PARAMS, alphas)

        msg.add(MPIMessage.MSG_ARG_KEY_LOCAL_TRAINING_ACC, valid_acc)
        msg.add(MPIMessage.MSG_ARG_KEY_LOCAL_TRAINING_LOSS, valid_loss)

        self.com_manager.send_message(msg)

    def __finish(self):
        logging.info("#######finished########### rank=%d", self.rank)
        self.com_manager.stop_receive_message()
        sys.exit()
