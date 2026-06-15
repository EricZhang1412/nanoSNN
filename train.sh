#!/bin/bash
# Single-device training script (Ascend NPU/CUDA/CPU auto-detected)
# Usage: bash train.sh [model_config] [data_config]
# Example: bash train.sh spikformer_tiny cifar10

MODEL=${1:-spikformer_tiny}
DATA=${2:-cifar10}
OPTIMIZER=${3:-sdtv3_cifar10}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
        PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
    else
        PYTHON_BIN=python
    fi
fi

export NANOSNN_ACCELERATOR=${NANOSNN_ACCELERATOR:-auto}
export NPU_PER_NODE=${NPU_PER_NODE:-1}

"${PYTHON_BIN}" train.py \
    --project_config configs/default_project_configs.yaml \
    --data_config    configs/data_configs/${DATA}.yaml \
    --train_config   configs/train_configs/default.yaml \
    --model_config   configs/model_configs/${MODEL}.yaml \
    --optimizer_config configs/optimizer_configs/${OPTIMIZER}.yaml \
    --resume auto
