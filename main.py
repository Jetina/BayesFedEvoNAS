# -*- coding: utf-8 -*-
import argparse
import logging
import os
import socket

import numpy as np
import psutil
import setproctitle
import torch
from mpi4py import MPI

from client.client_manager import ClientMananger
from data_preprocessing.data_loader import (
    partition_data,
    get_dataloader,
    get_dataloader1,
    get_dataloader2,
)
from server.server_manager import ServerMananger
from model.FedNASAggregator import FedNASAggregator
from model.FedNASTrainer import FedNASTrainer

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def add_args(parser: argparse.ArgumentParser):
    # ===== 基本 =====
    parser.add_argument("--gpu", type=int, default=1, help="number of gpus to use (for clients mapping)")
    parser.add_argument("--stage", type=str, default="search", help="search/train")

    # 兼容参数（你的 sh 里有 --model darts）
    parser.add_argument("--model", type=str, default="darts", help="(compat) not used, keep for old scripts")

    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="cifar10/mnist/fashionmnist/cinic10/medmnist/mydataLung")
    parser.add_argument("--datadir", type=str, default="./data")

    parser.add_argument("--partition", type=str, default="homo")   # homo / hetero
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3, help="Dirichlet alpha for hetero partition")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)

    # 重要：MPI 的 client 数 = size-1；同时这里也用于数据划分 client 数
    parser.add_argument("--client_number", type=int, default=10)
    parser.add_argument("--comm_round", type=int, default=50)

    parser.add_argument("--frequency_of_the_test", type=int, default=1)

    # ===== 分区固定随机种子（保证所有 rank 的 net_dataidx_map 一致）=====
    parser.add_argument("--partition_seed", type=int, default=0, help="seed used ONLY for data partitioning")

    # ===== 本地 train/val 划分比例（0.8/0.2）=====
    parser.add_argument("--local_train_ratio", type=float, default=0.8,
                        help="local train ratio, rest for local val (e.g., 0.8 -> 80/20)")

    # ===== DARTS/FedNAS =====
    parser.add_argument("--init_channels", type=int, default=16)
    parser.add_argument("--layers", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=0.025)
    parser.add_argument("--learning_rate_min", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=3e-4)

    parser.add_argument("--arch_learning_rate", type=float, default=3e-4)
    parser.add_argument("--arch_weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--lambda_train_regularizer", type=float, default=1.0)
    parser.add_argument("--lambda_valid_regularizer", type=float, default=1.0)
    parser.add_argument("--report_freq", type=int, default=10)

    parser.add_argument("--auxiliary", action="store_true", default=False)
    parser.add_argument("--auxiliary_weight", type=float, default=0.4)

    # ===== BayesFedEvoNAS =====
    parser.add_argument("--gp_train_round", type=int, default=35, help="evolution start round")
    parser.add_argument("--zeta", type=float, default=0.4)
    parser.add_argument("--epsilon", type=float, default=0.6)

    parser.add_argument("--arch_step_interval", type=int, default=1)

    parser.add_argument("--gp_warmup_samples", type=int, default=20)
    parser.add_argument("--gp_update_every", type=int, default=2)
    parser.add_argument("--gp_train_steps", type=int, default=1)
    parser.add_argument("--gp_inducing_points", type=int, default=16)
    parser.add_argument("--gp_lr", type=float, default=0.01)
    parser.add_argument("--gp_eval_batches", type=int, default=5)

    parser.add_argument("--elite_pool_size", type=int, default=50)
    parser.add_argument("--evo_generations", type=int, default=30)
    parser.add_argument("--evo_tournament_k", type=int, default=10)
    parser.add_argument("--evo_crossover_rate", type=float, default=0.9)
    parser.add_argument("--evo_mutate_rate", type=float, default=0.1)
    parser.add_argument("--evo_mutate_sigma", type=float, default=0.1)

    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--grad_accum_steps", type=int, default=1)

    # ===== 结果输出目录 =====
    parser.add_argument("--result_root", type=str, default="result", help="root dir for results/logs")

    return parser.parse_args()


def init_training_device(process_id: int, args):
    # server 固定 cuda:0
    if process_id == 0:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # clients 轮流映射到 [0, gpu-1]
    gpu_number = max(int(args.gpu), 1)
    client_index = process_id - 1
    gpu_index = client_index % gpu_number
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    logging.info("process %d -> device %s", process_id, str(device))
    return device


def _log_label_dist(tag: str, y):
    try:
        if isinstance(y, torch.Tensor):
            yy = y.detach().cpu().view(-1).numpy()
        else:
            yy = np.array(y).reshape(-1)
        uniq, cnt = np.unique(yy, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}
        maj = float(cnt.max() / cnt.sum()) if cnt.sum() > 0 else 0.0
        logging.info("[%s] label_dist=%s | majority_baseline=%.4f", tag, dist, maj)
    except Exception as e:
        logging.warning("[%s] label dist log failed: %s", tag, str(e))


def _stratified_local_split(dataidxs, y_train, train_ratio, seed):
    """
    在单个 client 的 dataidxs 内做分层切分（尽量保持train/val标签比例一致）。
    如果 client 只有一个类别，则退化为随机切分（保证可运行）。
    """
    dataidxs = np.asarray(dataidxs, dtype=int)
    n = int(len(dataidxs))
    if n < 2:
        return dataidxs, np.array([], dtype=int)

    train_ratio = float(train_ratio)
    train_ratio = min(max(train_ratio, 0.01), 0.99)

    rng = np.random.RandomState(int(seed))

    # y_train 保险转换成 numpy
    if isinstance(y_train, torch.Tensor):
        y_train_np = y_train.detach().cpu().view(-1).numpy()
    else:
        y_train_np = np.asarray(y_train).reshape(-1)

    y_local = y_train_np[dataidxs]
    classes = np.unique(y_local)

    # 只有一个类 -> 随机切分
    if len(classes) <= 1:
        perm = rng.permutation(dataidxs)
        split = int(np.floor(train_ratio * n))
        split = max(1, min(split, n - 1))
        return perm[:split], perm[split:]

    train_parts, val_parts = [], []
    for c in classes:
        idx_c = dataidxs[y_local == c]
        idx_c = rng.permutation(idx_c)
        n_c = len(idx_c)
        if n_c == 1:
            # 这个类只有1个样本：随机放一边（不可避免）
            if rng.rand() < train_ratio:
                train_parts.append(idx_c)
            else:
                val_parts.append(idx_c)
            continue

        split_c = int(np.floor(train_ratio * n_c))
        split_c = max(1, min(split_c, n_c - 1))
        train_parts.append(idx_c[:split_c])
        val_parts.append(idx_c[split_c:])

    train_idxs = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
    val_idxs = np.concatenate(val_parts) if val_parts else np.array([], dtype=int)

    # 极端情况下兜底：保证两边都非空
    if len(train_idxs) == 0 or len(val_idxs) == 0:
        perm = rng.permutation(dataidxs)
        split = int(np.floor(train_ratio * n))
        split = max(1, min(split, n - 1))
        train_idxs, val_idxs = perm[:split], perm[split:]

    train_idxs = rng.permutation(train_idxs)
    val_idxs = rng.permutation(val_idxs)
    return train_idxs, val_idxs


def init_server(args, comm, rank, size, round_num, device):
    logging.info("load dataset on server: %s", args.dataset)

    args_datadir = args.datadir
    args_logdir = f"log/{args.dataset}"

    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset,
        args_datadir,
        args_logdir,
        args.partition,
        args.client_number,
        args.dirichlet_alpha,
        args=args,
    )
    logging.info("traindata_cls_counts=%s", str(traindata_cls_counts))
    all_train_data_num = sum(len(net_dataidx_map[r]) for r in range(args.client_number))

    if args.dataset == "mydataLung":
        train_global = get_dataloader2(X_train, y_train, args.batch_size, shuffle=True)
        test_global = get_dataloader2(X_test, y_test, args.batch_size, shuffle=False)
        _log_label_dist("SERVER_GLOBAL_TEST", y_test)
    else:
        train_global, test_global = get_dataloader(args.dataset, args_datadir, args.batch_size, args.batch_size)

    client_num = size - 1
    aggregator = FedNASAggregator(train_global, test_global, all_train_data_num, client_num, device, args)
    server_manager = ServerMananger(args, comm, rank, size, round_num, aggregator)
    server_manager.run()


def init_client(args, comm, rank, size, round_num, seed, device):
    """
    seed 只用于训练随机性（batch/shuffle/初始化等）。
    partition_data() 的分区随机性由 args.partition_seed 控制，独立。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    client_id = rank - 1
    logging.info("load dataset on client %d: %s", client_id, args.dataset)

    args_datadir = args.datadir
    args_logdir = f"log/{args.dataset}"

    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset,
        args_datadir,
        args_logdir,
        args.partition,
        args.client_number,
        args.dirichlet_alpha,
        args=args,
    )

    all_train_data_num = sum(len(net_dataidx_map[r]) for r in range(args.client_number))
    dataidxs = net_dataidx_map[client_id]
    local_sample_number = len(dataidxs)

    # ===== 本地分层 0.8/0.2 划分 =====
    train_ratio = float(getattr(args, "local_train_ratio", 0.8))
    # 用“client_id + partition_seed”保证：同一划分下每个client的split固定可复现
    split_seed = int(getattr(args, "partition_seed", 0)) * 100000 + int(client_id)
    train_idxs, val_idxs = _stratified_local_split(dataidxs, y_train, train_ratio, split_seed)

    if args.dataset == "mydataLung":
        train_local = get_dataloader1(X_train, y_train, args.batch_size, train_idxs, shuffle=True)
        test_local = get_dataloader1(X_train, y_train, args.batch_size, val_idxs, shuffle=False)
    else:
        train_local, _ = get_dataloader(args.dataset, args_datadir, args.batch_size, args.batch_size, train_idxs)
        test_local, _ = get_dataloader(args.dataset, args_datadir, args.batch_size, args.batch_size, val_idxs)

    logging.info("client=%d local_sample_number=%d train=%d val=%d (ratio=%.2f)",
                 client_id, local_sample_number, len(train_idxs), len(val_idxs), train_ratio)

    trainer = FedNASTrainer(client_id, train_local, test_local, local_sample_number, all_train_data_num, device, args)
    client_manager = ClientMananger(args, comm, rank, size, round_num, trainer)
    client_manager.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    setproctitle.setproctitle("Federated Learning:" + str(rank))
    logging.basicConfig(
        level=logging.INFO,
        format=str(rank) + " - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
    )

    hostname = socket.gethostname()
    logging.info("process=%d host=%s pid=%d proc=%s",
                 rank, hostname, os.getpid(), str(psutil.Process(os.getpid())))

    if size != args.client_number + 1:
        logging.warning("MPI size=%d, but args.client_number=%d. Expected np = client_number + 1 = %d",
                        size, args.client_number, args.client_number + 1)

    base_seed = 12345
    client_seed = base_seed + rank * 97 + (os.getpid() % 997)

    device = init_training_device(rank, args)
    round_num = args.comm_round

    if rank == 0:
        init_server(args, comm, rank, size, round_num, device)
    else:
        init_client(args, comm, rank, size, round_num, client_seed, device)
