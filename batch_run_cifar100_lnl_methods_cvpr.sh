#!/bin/bash

# 检查是否传递了GPU的参数
if [ -z "$1" ]; then
    echo "使用方法: ./this_script.sh <GPU_ID>"
    exit 1
fi

# 设置指定的 GPU
export CUDA_VISIBLE_DEVICES=$1
echo "Using GPU $1"
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"

# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

sh run_cifar100_lnl_coteaching_cvpr.sh $CUDA_VISIBLE_DEVICES       # Coteaching
sh run_cifar100_lnl_coteaching_plus_cvpr.sh $CUDA_VISIBLE_DEVICES  # Coteachingplus
sh run_cifar100_lnl_jocor_cvpr.sh $CUDA_VISIBLE_DEVICES            # JoCoR
sh run_cifar100_lnl_decoupling_cvpr.sh $CUDA_VISIBLE_DEVICES       # Decoupling， 200 epochs
sh run_cifar100_lnl_negativeLearning_cvpr.sh $CUDA_VISIBLE_DEVICES # NegativeLearning， 200 epochs
sh run_cifar100_lnl_pencil_cvpr.sh $CUDA_VISIBLE_DEVICES           # PENCIL, 200 epochs
