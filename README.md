# nanoSNN

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.x-792ee5.svg)](https://lightning.ai/)
[![uv](https://img.shields.io/badge/dependencies-uv-6e56cf.svg)](https://github.com/astral-sh/uv)
[![GitHub stars](https://img.shields.io/github/stars/EricZhang1412/nanoSNN?style=social)](https://github.com/EricZhang1412/nanoSNN)

nanoSNN is a general vision SNN training framework inspired by `nanoLMengine`.
It provides a config-driven training workflow for representative spiking vision models, with a unified `train.py` entrypoint, Lightning-based training loop, checkpoint resume support, and reusable dataset/model building blocks.

## Recent News

- 🎉🎉🎉**2026-05-04**: Implement Billeh-v1 visual cortex-inspired model. 
> Ref: 
> - 📑[A data-based large-scale model for primary visual cortex enables brain-like robust and versatile visual processing](https://www.science.org/doi/10.1126/sciadv.abq7592); 
> - 📁[Orignial Code in Tensorflow](https://github.com/ifgovh/Training-data-driven-V1-model); 
> - 📁[Naive Implementation in Pytorch](https://github.com/EricZhang1412/Training-data-driven-V1-model).

## Overview

This project currently supports:

- **Spikformer**
- **Spike-Driven Transformer** v1 / v2 / v3
- **Spiking CNNs**: VGG / ResNet / MS-ResNet
- **Static-image datasets**: CIFAR10 / CIFAR100 / ImageNet
- **Event-data extension hooks**: CIFAR10-DVS / DVS128 Gesture

The framework follows a multi-YAML configuration style similar to `nanoLMengine`:

- `project_config`
- `data_config`
- `train_config`
- `model_config`
- `optimizer_config`

It is designed to make it easy to:

- switch models and datasets with config files
- keep a consistent training pipeline across model families
- support both static and event-based visual inputs
- extend the project with additional SNN architectures later

## Features

- Config-driven training workflow
- Unified `train.py` entrypoint
- PyTorch Lightning trainer setup
- Checkpoint save/resume support (`auto | none | path`)
- WandB / TensorBoard logging
- Automatic static-image expansion from `[B, C, H, W]` to `[T, B, C, H, W]`
- Native temporal input support for event data
- Automatic `reset_net` after each forward pass to avoid cross-batch state leakage

## Environment Setup

### Requirements

Recommended environment:

- Python `>=3.10,<3.13`
- CUDA-compatible PyTorch environment if training on GPU
- [`uv`](https://github.com/astral-sh/uv) for dependency management

### Install dependencies

From the project root:

```bash
uv sync
```

Then run commands through `uv run`.

## Project Structure

```text
nanoSNN/
├── configs/
│   ├── data_configs/
│   ├── model_configs/
│   ├── optimizer_configs/
│   └── train_configs/
├── data/
├── models/
├── utils/
├── train.py
├── train.sh
└── multigpu_train.sh
```

## Usage Examples

### 1. Single-GPU training

Run with default settings:

```bash
uv run bash train.sh
```

Run a specific model on a specific dataset:

```bash
uv run bash train.sh spikformer_tiny cifar10
uv run bash train.sh spiking_resnet18 cifar100
```

### 2. Multi-GPU training

Example:

```bash
uv run bash multigpu_train.sh spiking_resnet18 imagenet 2 1
```

Arguments:

```bash
bash multigpu_train.sh [model_config] [data_config] [gpus_per_node] [num_nodes]
```

### 3. Manual training command

```bash
uv run python train.py \
  --project_config configs/default_project_configs.yaml \
  --data_config configs/data_configs/cifar10.yaml \
  --train_config configs/train_configs/default.yaml \
  --model_config configs/model_configs/spikformer_tiny.yaml \
  --optimizer_config configs/optimizer_configs/default.yaml \
  --resume auto
```

## Configuration

### Project config

Example field:

- `output_dir`: root directory for logs, checkpoints, and outputs

### Data config

Common fields:

- `name`
- `root`
- `num_classes`
- `image_size`
- `in_channels`
- `is_event`
- `num_workers`
- `pin_memory`

### Train config

Common fields:

- `batch_size_per_gpu`
- `random_seed`
- `trainer.strategy`
- `trainer.precision`
- `trainer.max_epochs`
- `trainer.accumulate_grad_batches`
- `trainer.save_every_n_train_steps`

### Model config

Common fields:

- `name`
- `T`
- `num_classes`
- `image_size`
- `in_channels`
- `neuron_type`
- `tau`
- `v_threshold`
- `detach_reset`

Additional architecture-specific fields:

- Transformer models: `embed_dim`, `depth`, `num_heads`, `mlp_ratio`, `patch_size`
- CNN models: `variant` or `arch`, `dropout`

### Optimizer config

Common fields:

- `lr`
- `beta1`, `beta2`
- `weight_decay`
- `scheduler`
- `warmup_steps`
- `min_lr_ratio`

## Available Model Configs

Currently included:

- `spikformer_tiny`
- `sdt_v1_small`
- `sdt_v2_small`
- `sdt_v3_small`
- `spiking_vgg11`
- `spiking_resnet18`
- `ms_resnet18`

These are located in:

```text
configs/model_configs/
```

## Data Convention

- Static image input: `[B, C, H, W]`
- Expanded internally to temporal input: `[T, B, C, H, W]`
- Event input: `[T, B, C, H, W]`
- Model output: classification logits, aggregated to `[B, num_classes]`

## Logging and Checkpoints

- Single-GPU runs use `WandbLogger`
- Multi-GPU runs use `TensorBoardLogger`
- Checkpoints are stored under:

```text
<output_dir>/checkpoints/<exp_name>/
```

Resume modes:

- `--resume auto`
- `--resume none`
- `--resume /path/to/xxx.ckpt`

## Example Training Setups

### CIFAR10 + Spikformer

```bash
uv run bash train.sh spikformer_tiny cifar10
```

### CIFAR100 + SDT v1

```bash
uv run python train.py \
  --project_config configs/default_project_configs.yaml \
  --data_config configs/data_configs/cifar100.yaml \
  --train_config configs/train_configs/default.yaml \
  --model_config configs/model_configs/sdt_v1_small.yaml \
  --optimizer_config configs/optimizer_configs/default.yaml \
  --resume auto
```

### ImageNet + Spiking ResNet

```bash
uv run bash multigpu_train.sh spiking_resnet18 imagenet 8 1
```

## Notes

- This is a first version focused on a unified training framework plus representative model families.
- Event-dataset configs are already included as extension points for future training support.
- Before training, update dataset paths in the YAML configs to match your local environment.
