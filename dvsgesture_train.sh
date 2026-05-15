#!/bin/bash
  # Multi-GPU DDP training: billeh-v1 + LGN on DVS128Gesture.
  # 128x128 spatial -> ~7x memory of CIFAR10-DVS; need smaller per-GPU batch.                                                            
  # Usage:                                                                                                                               
  #   bash multigpu_train_dvsgesture.sh                                                                                                  
  #   bash multigpu_train_dvsgesture.sh 4 1 29504                                                                                        

GPUS=${1:-2}
NODES=${2:-1}
PORT=${3:-29504}
export GPU_PER_NODE=${GPUS}
export N_NODE=${NODES}

torchrun \
    --nproc_per_node=${GPUS} \
    --nnodes=${NODES} \
    --master_port=${PORT} \
    train.py \
    --project_config   configs/default_project_configs.yaml \
    --data_config      configs/data_configs/dvs128gesture_direct.yaml \
    --train_config     configs/train_configs/transformer_dvs128gesture.yaml \
    --model_config     configs/model_configs/billeh_v1_dvs128gesture_direct.yaml \
    --optimizer_config configs/optimizer_configs/billeh_dvs_direct.yaml \
    --resume none        