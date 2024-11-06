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

# run_cifar10_crul_cvpr.sh
python ./core_model/CRUL.py --model cifar-resnet18 --dataset cifar-10 --num_epochs 3 --repair_iter_num 5 --uni_name CRUL --learning_rate 1e-3 --optimizer adam --batch_size 256 --noise_ratio 0.25

# run_flower102_crul_cvpr.sh
python ./core_model/CRUL.py --model wideresnet50 --dataset flower-102 --num_epochs 3 --repair_iter_num 5 --uni_name CRUL --learning_rate 1e-4 --optimizer adam --batch_size 256 --noise_type symmetric --noise_ratio 0.5

# run_cifar100_crul_cvpr.sh
python ./core_model/CRUL.py --model cifar-wideresnet40 --dataset cifar-100 --repair_iter_num 5 --uni_name CRUL --learning_rate 1e-5 --optimizer adam --batch_size 256 --noise_type asymmetric --noise_ratio 0.25 --num_epochs 3

# run_pet37_crul_cvpr.sh
python ./core_model/CRUL.py --model wideresnet50 --dataset pet-37 --num_epochs 3 --repair_iter_num 5 --uni_name CRUL --learning_rate 2e-5 --optimizer adam --batch_size 64 --noise_type asymmetric --noise_ratio 0.25 --mixup_alpha 0.5
