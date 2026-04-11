#!/bin/bash
# Single-GPU training script
# Usage: bash train.sh [model_config] [data_config]
# Example: bash train.sh spikformer_tiny cifar10


MODEL=${1:-mem_gated_attn_tiny}
DATA=${2:-cifar10}
TRAIN=${3:-default}
OPTIMIZER=${4:-mem_gated_attn_tiny_cifar10}
PROJECT=${5:-smoke_test}



python train.py \
    --project_config configs/${PROJECT}.yaml \
    --data_config    configs/data_configs/${DATA}.yaml \
    --train_config   configs/train_configs/${TRAIN}.yaml \
    --model_config   configs/model_configs/${MODEL}.yaml \
    --optimizer_config configs/optimizer_configs/${OPTIMIZER}.yaml \
    --resume none
