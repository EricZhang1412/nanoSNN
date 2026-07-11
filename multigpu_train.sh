#!/bin/bash
# Multi-device / multi-node training script (DDP via torchrun)
# Usage: bash multigpu_train.sh [project] [train] [model] [data] [optimizer] [devices_per_node] [num_nodes] [port]
# Example: ASCEND_RT_VISIBLE_DEVICES=0,1 bash multigpu_train.sh default_project_configs default sdt_v1_small cifar10 sdtv1_cifar10 2 1

PROJECT=${1:-default_project_configs}
TRAINING=${2:-default}
MODEL=${3:-sdt_v1_small}
DATA=${4:-cifar10}
OPTIMIZER=${5:-sdtv1_cifar10}

DEVICES=${6:-1}
NODES=${7:-1}
PORT=${8:-29502}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${TORCHRUN_BIN:-}" ]; then
    if [ -x "${ROOT_DIR}/.venv/bin/torchrun" ]; then
        TORCHRUN_BIN="${ROOT_DIR}/.venv/bin/torchrun"
    else
        TORCHRUN_BIN=torchrun
    fi
fi

export NANOSNN_ACCELERATOR=${NANOSNN_ACCELERATOR:-auto}
export DEVICES_PER_NODE=${DEVICES}
export NPU_PER_NODE=${DEVICES}
export GPU_PER_NODE=${DEVICES}
export N_NODE=${NODES}

"${TORCHRUN_BIN}" \
    --nproc_per_node=${DEVICES} \
    --nnodes=${NODES} \
    --master_port=${PORT} \
    train.py \
    --project_config configs/${PROJECT}.yaml \
    --data_config    configs/data_configs/${DATA}.yaml \
    --train_config   configs/train_configs/${TRAINING}.yaml \
    --model_config   configs/model_configs/${MODEL}.yaml \
    --optimizer_config configs/optimizer_configs/${OPTIMIZER}.yaml \
    --resume auto
