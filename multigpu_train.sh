#!/bin/bash
# Multi-GPU / multi-node training script (DDP via torchrun)
# Usage: bash multigpu_train.sh [model_config] [data_config] [gpus_per_node] [num_nodes]
# Example: bash multigpu_train.sh spiking_resnet18 imagenet 8 1
export CUDA_VISIBLE_DEVICES=3,4,5,6

PROJECT=${1:-default_project_configs}
TRAINING=${2:-transformer}
MODEL=${3:-billeh_v1_seqcifar10}
DATA=${4:-sequential_cifar10_fewshot}
OPTIMIZER=${5:-transformer}

GPUS=${6:-4}
NODES=${7:-1}
PORT=${8:-29502}

export GPU_PER_NODE=${GPUS}
export N_NODE=${NODES}

torchrun \
    --nproc_per_node=${GPUS} \
    --nnodes=${NODES} \
    --master_port=${PORT} \
    train.py \
    --project_config configs/${PROJECT}.yaml \
    --data_config    configs/data_configs/${DATA}.yaml \
    --train_config   configs/train_configs/${TRAINING}.yaml \
    --model_config   configs/model_configs/${MODEL}.yaml \
    --optimizer_config configs/optimizer_configs/${OPTIMIZER}.yaml \
    --resume auto
