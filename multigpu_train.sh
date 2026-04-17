#!/bin/bash
# Multi-GPU / multi-node training script (DDP via torchrun)
# Usage: bash multigpu_train.sh [model_config] [data_config] [gpus_per_node] [num_nodes]
# Example: bash multigpu_train.sh spiking_resnet18 imagenet 8 1
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4
MODEL=${1:-mem_gated_attn_tiny}
DATA=${2:-imagenet_hf}
GPUS=${3:-1}
NODES=${4:-1}
OPTIMIZER=${5:-mem_gated_attn_tiny_imagenet1k}

export GPU_PER_NODE=${GPUS}
export N_NODE=${NODES}

torchrun \
    --nproc_per_node=${GPUS} \
    --nnodes=${NODES} \
    train.py \
    --project_config configs/default_project_configs.yaml \
    --data_config    configs/data_configs/${DATA}.yaml \
    --train_config   configs/train_configs/imagenet_hf.yaml \
    --model_config   configs/model_configs/${MODEL}.yaml \
    --optimizer_config configs/optimizer_configs/${OPTIMIZER}.yaml \
    --resume auto
