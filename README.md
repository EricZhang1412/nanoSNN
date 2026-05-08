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

## Transformer (Attention Is All You Need) baseline

This repo also includes a plain Transformer encoder baseline (ViT-style patch embedding + encoder stack),
implemented under `models/transformer/` and registered as `name: vit` / `name: transformer_vit`.

Model config example: `configs/model_configs/vit_tiny.yaml`.

## LRA: Pathfinder32 (hard)

To run the LRA Pathfinder32-hard task, we use a lightweight `npz` dataset format to avoid TFDS at training time.

### 1. Prepare dataset

This step downloads/parses TFDS once and writes:

```text
datasets/lra_pathfinder32_hard/
  train.npz
  val.npz
  test.npz
```

Generate the splits:

```bash
uv run python scripts/prepare_lra_pathfinder_npz.py \
  --out_root ./datasets/lra_pathfinder32_hard \
  --tfds_name pathfinder32
```

### 2. Train

Use the provided configs:

```bash
uv run python train.py \
  --project_config configs/default_project_configs.yaml \
  --data_config configs/data_configs/lra_pathfinder32_hard.yaml \
  --train_config configs/train_configs/lra_pathfinder.yaml \
  --model_config configs/model_configs/vit_tiny.yaml \
  --optimizer_config configs/optimizer_configs/lra_pathfinder.yaml \
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

## Strict Reproduction: Chen-Scherr-Maass V1+LGN MNIST

The `billeh_v1` model implements a strict-reproduction pipeline for the MNIST
experiment in Chen, Scherr, Maass, *A Data-Based Large-Scale Model for Primary
Visual Cortex Enables Brain-like Robust and Versatile Visual Processing*,
Sci. Adv. 8, eabq7592 (2022). The settings are aligned with the upstream
implementation at [ifgovh/Training-data-driven-V1-model](https://github.com/ifgovh/Training-data-driven-V1-model).

### Aligned settings

- Trial timing: `T = 600 ms`, `dt = 1 ms`, `pre_delay = 50`, `post_delay = 450`
  (image stimulus shown for `T - pre - post = 100 ms`).
- LGN front-end: 17,400-cell time-space filter bank on a 240×120 retinotopic field.
- Pixel scaling: `(x - 0.5) * sti_intensity / 0.5` with `sti_intensity = 2.0`.
- Neuron model: GLIF3 (membrane + 2 after-spike currents) with Billeh
  initialization from Allen Institute V1 connectivity (Dale's law enforced).
- Readout: 10 fixed L5e pools (pool ids 5..14, 30 neurons each), pool firing
  rates passed through "spike straight-through" gradient (`1/df` scaling) and
  binned to 50-ms response chunks; classification loss applied within the
  response window only.
- Loss: `CE_response + α_v · voltage_loss + α_f · rate_huber_quantile_loss`,
  with `α_v = 1e-5`, `α_f = 0.1`, Huber `κ = 0.002`.
- Voltage regularization: penalty on normalized voltage outside [-1, +1].
- Rate regularization: Huber-quantile distribution match to per-neuron target
  rates resampled by empirical percentile from `garrett_firing_rates.pkl`.
- Surrogate gradient: Gaussian (`σ = 0.5`, `dampening_factor = 0.3`).

### Known deviations (intentional simplifications)

| Item | Paper | This repo | Why |
|------|-------|-----------|-----|
| Optimizer | SGD + momentum, BPTT | AdamW + cosine, BPTT | Stable on single 4090; minor |
| GPUs | 160 GPUs × 60 hours | 1× RTX 4090 24GB | Hardware budget |
| `n_neurons` | 51,798 (full core) | 10,000–20,000 (random sub-core) | Memory budget |
| `neurons_per_output` | 30 per spatially-localized L5 pool | 8–16 random L5e per pool (`localized_readout: false`) | Sub-core L5e density (~1500 at N=10k) is too low to populate spatial pools at radius 55; fall back to random L5e sampling |
| LGN→V1 input units | Hz × W_in (continuous current) | (Hz / 1000) × W_in via `lgn_input_scale=1e-3` | Original code calibrates W_in for spike-per-ms inputs; raw Hz over-injects 1000× |
| Precision | fp32 | bf16-mixed (fallback fp32) | Memory; falls back if NaN |
| Tasks | 5-task joint training (320 trials/batch) | Single-task MNIST (effective batch 64 via accumulation) | This repo focuses on single-task strict reproduction |
| Epochs | 16 (joint) | 16 (single-task) | Same training compute order |

### Required data files

Under `model_config.billeh_data_dir` (default `/data2/dataset/LGN_GLIF_Models/preprocessed/GLIF_network`):

- `network_dat.pkl`, `v1_node_types.csv`, `v1_nodes.h5` (Billeh GLIF network).
- `garrett_firing_rates.pkl` (per-cell firing rates from Allen experimental data,
  used as the target rate distribution).

Under `model_config.lgn_data_path` directory (default `/data2/dataset/LGN_GLIF_Models/new_0505/GLIF_network2`):

- `lgn_full_col_cells_3.csv`, `temporal_kernels.pkl`.

Generate `temporal_kernels.pkl` once with:

```bash
uv run python scripts/prepare_lgn_kernels.py --lgn_py /path/to/upstream/lgn_model/lgn.py
```

### Quick start

```bash
# Smoke test (B=2, single fwd+bwd, no real data needed beyond Billeh + LGN files)
uv run python -m scripts.smoke_billeh_v1_lgn --batch 2

# Memory check at full strict-repro batch
uv run python -m scripts.smoke_billeh_v1_lgn --batch 8 --memory

# Full training run
uv run python train.py \
  --project_config configs/default_project_configs.yaml \
  --data_config configs/data_configs/billeh_mnist_lgn.yaml \
  --train_config configs/train_configs/billeh_mnist.yaml \
  --model_config configs/model_configs/billeh_v1_mnist_lgn.yaml \
  --optimizer_config configs/optimizer_configs/billeh_mnist.yaml \
  --resume auto
```

### Memory budget

`T=600`, `n_neurons=10000`, `chunk_size=60` (10 chunks via gradient
checkpointing), `batch_size=8`, `accumulate_grad_batches=8` → effective batch 64.
Peak GPU memory should fit under ~22 GiB. If OOM, reduce `chunk_size` to 30 or
`batch_size` to 4 (with `accumulate_grad_batches=16`). If `bf16-mixed` produces
NaN inside the first 100 steps, switch `precision` to `"32-true"`.

## Notes

- This is a first version focused on a unified training framework plus representative model families.
- Event-dataset configs are already included as extension points for future training support.
- Before training, update dataset paths in the YAML configs to match your local environment.
