#!/bin/bash

# 检查是否传递了GPU的参数
if [ -z "$1" ]; then
    echo "使用方法: ./run_cifar_replay.sh <GPU_ID>"
    exit 1
fi

# 设置指定的 GPU
export CUDA_VISIBLE_DEVICES=$1
echo "Using GPU $1"
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"

# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

python ./run_experiment.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --step 0 --learning_rate 0.0001 --optimizer adam --batch_size 256 --uni_name replay --balanced --noise_type asymmetric --model_suffix worker_restore

python ./run_experiment.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --step 1 --learning_rate 0.0001 --optimizer adam --batch_size 256 --uni_name replay --balanced --noise_type asymmetric --model_suffix worker_restore

python ./run_experiment.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --step 2 --learning_rate 0.0001 --optimizer adam --batch_size 256 --uni_name replay --balanced --noise_type asymmetric --model_suffix worker_restore

python ./run_experiment.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --step 3 --learning_rate 0.0001 --optimizer adam --batch_size 256 --uni_name replay --balanced --model_suffix worker_restore
