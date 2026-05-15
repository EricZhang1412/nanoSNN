#!/bin/bash                                                                                                                            
# Multi-GPU DDP training: billeh-v1 + LGN on CIFAR10-DVS.                                                                              
# Usage:                                                                                                                               
#   bash multigpu_train_cifar10dvs.sh                # default: 4 GPUs, 1 node                                                         
#   bash multigpu_train_cifar10dvs.sh 8              # 8 GPUs                                                                          
#   bash multigpu_train_cifar10dvs.sh 4 1 29503      # 4 GPUs, 1 node, port 29503                                                      
GPUS=${1:-6}                                                                                                                           
NODES=${2:-1}
PORT=${3:-29502}
export GPU_PER_NODE=${GPUS}
export N_NODE=${NODES}

torchrun \
    --nproc_per_node=${GPUS} \
    --nnodes=${NODES} \
    --master_port=${PORT} \
    train.py \
    --project_config configs/default_project_configs.yaml \
    --data_config configs/data_configs/cifar10dvs_lgn.yaml \
    --train_config configs/train_configs/transformer.yaml \
    --model_config configs/model_configs/billeh_v1_cifar10dvs_direct.yaml \
    --optimizer_config configs/optimizer_configs/billeh_dvs_direct.yaml \
    --resume none