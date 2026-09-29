# BayesFedEvoNAS: Federated Neural Architecture Search with SGP-Guided Evolution and Prediction-Driven Aggregation

This repository contains the source code for the paper:

> **BayesFedEvoNAS: Federated Neural Architecture Search with SGP-Guided Evolution and Prediction-Driven Aggregation**\
> Jie Tian, Xin Liu, Qiqi Liu, Huizhi Liu, Xuechen Zhao

## 1. Overview

BayesFedEvoNAS is a federated neural architecture search (FedNAS) framework that improves both ends of the federated search loop:

- **SGP-Guided Evolution (client side).** During local architecture search, each client builds a sparse Gaussian process (SGP) surrogate over candidate architectures and uses it to guide an evolutionary local search, so that promising architectures are identified with fewer local evaluations.
- **Prediction-Driven Aggregation (server side).** Instead of naively averaging architecture parameters (alphas), the server aggregates client updates according to their predicted accuracies, giving more weight to clients whose architectures are expected to perform better.

The system follows the two-component design of [FedNAS](https://github.com/RuiTong1998/FedNAS) (communication protocol + on-device learning), with the search and aggregation logic replaced by the methods above. The supernet and search space are inherited from [DARTS](https://github.com/quark0/darts).

## 2. Repository Structure

```
.
├── main.py                        # Entry point (MPI-launched)
├── run_fednas_search.sh           # Search stage launcher
├── run_fednas_train.sh            # Train stage launcher
├── server/server_manager.py       # Server-side manager
├── client/client_manager.py       # Client-side manager
├── model/
│   ├── FedNASTrainer.py           # Client: local search/training with SGP-guided evolution
│   ├── FedNASAggregator.py        # Server: FedAvg + prediction-driven aggregation
│   ├── local_search.py            # Local (evolutionary) search variant
│   └── origin.py                  # Original FedNAS aggregator (baseline)
├── darts/                         # DARTS supernet, operations, genotypes
├── data_preprocessing/datasets.py # Dataset partitioning (homo/hetero)
├── communication/                 # MPI-based communication
└── topology/                      # Topology manager
```

## 3. Requirements

- Python 3.7+
- PyTorch (tested with 1.4.0+)
- [mpi4py](https://pypi.org/project/mpi4py/) 3.0.3
- [gpytorch](https://gpytorch.ai/) (sparse GP surrogate)
- [thop](https://github.com/Lyken17/pytorch-OpCounter) (FLOPs/params counting)
- [wandb](https://www.wandb.com/) (experiment tracking, optional)

```bash
pip install torch mpi4py gpytorch thop wandb
```

## 4. Usage

**Stage 1: Architecture search**

```bash
bash run_fednas_search.sh <GPU> <MODEL> <homo|hetero> <ROUND> <EPOCH> <BATCH_SIZE>
```

This launches `mpirun -np 15` (1 server + 14 clients) on CIFAR-10.

**Stage 2: Train the discovered architecture**

```bash
bash run_fednas_train.sh <GPU> <MODEL> <homo|hetero> <ROUND> <EPOCH> <BATCH_SIZE>
```

Edit the scripts (client number, dataset, host file) to match your cluster setup.

## 5. Acknowledgments

- [FedNAS: Federated Deep Learning via Neural Architecture Search](https://chaoyanghe.com/publications/FedNAS-CVPR2020-NAS.pdf) (He, Annavaram, Avestimehr; CVPR 2020 Workshop on NAS) — base system design.
- [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) (Liu et al., ICLR 2019) — search space and supernet implementation.

## 6. Citation

If you find this code useful, please cite:

```bibtex
@article{tian2026bayesfedevonas,
  title   = {BayesFedEvoNAS: Federated Neural Architecture Search with SGP-Guided Evolution and Prediction-Driven Aggregation},
  author  = {Jie Tian and Xin Liu and Qiqi Liu and Huizhi Liu and Xuechen Zhao},
  year    = {2026}
}
```
