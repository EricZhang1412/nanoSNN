#!/bin/bash
# Single-GPU training script
# Usage: bash train.sh [model_config] [data_config]
# Example: bash train.sh spikformer_tiny cifar10

MODEL=${1:-spikformer_tiny}
DATA=${2:-cifar10}

python train.py \
    --project_config configs/default_project_configs.yaml \
    --data_config    configs/data_configs/${DATA}.yaml \
    --train_config   configs/train_configs/default.yaml \
    --model_config   configs/model_configs/${MODEL}.yaml \
    --optimizer_config configs/optimizer_configs/default.yaml \
    --resume none
