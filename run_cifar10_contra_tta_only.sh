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

python ./run_experiment.py --model cifar-resnet18 --dataset cifar-10 --num_epochs 10 --step 0 --learning_rate 0.01 --optimizer adam --batch_size 256 --balanced --uni_name contra_tta_only

python ./core_model/train_teacher.py --model cifar-resnet18 --dataset cifar-10 --num_epochs 50 --step 0 --learning_rate 0.005 --optimizer adam --batch_size 256 --balanced  --uni_name contra_tta_only

python ./run_experiment.py --model cifar-resnet18 --dataset cifar-10 --num_epochs 5 --step 1 --learning_rate 0.001 --optimizer adam  --batch_size 256 --balanced --model_suffix worker_restore --uni_name contra_tta_only

python ./run_experiment.py --model cifar-resnet18 --dataset cifar-10 --num_epochs 5 --step 2 --learning_rate 0.001 --optimizer adam  --batch_size 256 --balanced --model_suffix worker_restore --uni_name contra_tta_only

python ./run_experiment.py --model cifar-resnet18 --dataset cifar-10 --num_epochs 5 --step 3 --learning_rate 0.001 --optimizer adam  --batch_size 256 --balanced --model_suffix worker_restore --uni_name contra_tta_only

python ./core_model/core.py --model cifar-resnet18 --dataset cifar-10 --num_epochs 2 --step 1 --learning_rate 0.0005 --optimizer adam --batch_size 256 --balanced --tta_only 0 --uni_name contra_tta_only

python ./core_model/core.py --model cifar-resnet18 --dataset cifar-10 --num_epochs 2 --step 2 --learning_rate 0.0005 --optimizer adam --batch_size 256 --balanced --tta_only 0 --uni_name contra_tta_only

python ./core_model/core.py --model cifar-resnet18 --dataset cifar-10 --num_epochs 2 --step 3 --learning_rate 0.0005 --optimizer adam --batch_size 256 --balanced --tta_only 0 --uni_name contra_tta_only