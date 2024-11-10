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

python ./run_experiment_cvpr.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --train_mode finetune --learning_rate 1e-4 --optimizer adam --batch_size 16 --noise_type asymmetric --noise_ratio 0
# python ./run_experiment_cvpr.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --train_mode finetune --learning_rate 1e-4 --optimizer adam --batch_size 16 --noise_type asymmetric --noise_ratio 0.25
python ./run_experiment_cvpr.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --train_mode finetune --learning_rate 1e-4 --optimizer adam --batch_size 16 --noise_type asymmetric --noise_ratio 0.5
python ./run_experiment_cvpr.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --train_mode finetune --learning_rate 1e-4 --optimizer adam --batch_size 16 --noise_type asymmetric --noise_ratio 0.75