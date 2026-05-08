#!/bin/bash
# Single-GPU training script
# Usage: bash train.sh [model_config] [data_config]
# Example: bash train.sh spikformer_tiny cifar10

python train.py \
  --project_config configs/default_project_configs.yaml \
  --data_config configs/data_configs/lra_cifar10.yaml \
  --train_config configs/train_configs/lra_cifar10_transformer.yaml \
  --model_config configs/model_configs/lra_transformer_cifar10.yaml \
  --optimizer_config configs/optimizer_configs/lra_cifar10_transformer.yaml \
  --resume auto