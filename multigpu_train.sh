#!/bin/bash
# Multi-GPU / multi-node training script (DDP via torchrun)
# Usage: bash multigpu_train.sh [model_config] [data_config] [gpus_per_node] [num_nodes]
# Example: bash multigpu_train.sh spiking_resnet18 imagenet 8 1

MODEL=${1:-spikformer_tiny}
DATA=${2:-cifar10}
GPUS=${3:-2}
NODES=${4:-1}

export GPU_PER_NODE=${GPUS}
export N_NODE=${NODES}

torchrun \
    --nproc_per_node=${GPUS} \
    --nnodes=${NODES} \
    train.py \
    --project_config configs/default_project_configs.yaml \
    --data_config    configs/data_configs/${DATA}.yaml \
    --train_config   configs/train_configs/default.yaml \
    --model_config   configs/model_configs/${MODEL}.yaml \
    --optimizer_config configs/optimizer_configs/default.yaml \
    --resume auto
