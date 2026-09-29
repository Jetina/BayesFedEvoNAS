#!/usr/bin/env bash

GPU=$1
MODEL=$2
# homo; hetero
DISTRIBUTION=$3
ROUND=$4
EPOCH=$5
BATCH_SIZE=$6

mpirun -np 15 python3 ./main.py \
  --gpu $GPU \
  --model $MODEL \
  --dataset cifar10 \
  --partition $DISTRIBUTION  \
  --client_number 14 \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE