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

python ./baseline_code/lips-mu/main_mu_cvpr.py --model cifar-wideresnet40 --dataset cifar-100 --num_epochs 10 --batch_size 256 --uni_name FT_prune_bi --unlearn_lr 1e-3 --noise_type asymmetric --noise_ratio 0.25 --alpha 1e-5

# python ./run_experiment.py --model cifar-wideresnet40 --dataset cifar-100 --num_epochs 10 --step 0 --batch_size 32 --balanced --uni_name JoCoR --learning_rate 5e-3

# python ./run_experiment.py --model cifar-wideresnet40 --dataset cifar-100 --num_epochs 10 --step 1 --batch_size 32 --balanced --model_suffix worker_raw --uni_name JoCoR --learning_rate 5e-3

# python baseline_code/colearn/main.py --model cifar-wideresnet40 --dataset cifar-100 --num_epochs 10 --step 1 --batch_size 32 --balanced --uni_name JoCoR --learning_rate 5e-3

# python ./run_experiment.py --model cifar-wideresnet40 --dataset cifar-100 --num_epochs 10 --step 2 --batch_size 32 --balanced --model_suffix worker_raw --uni_name JoCoR --learning_rate 5e-3

# python baseline_code/colearn/main.py --model cifar-wideresnet40 --dataset cifar-100 --num_epochs 10 --step 2 --batch_size 32 --balanced --uni_name JoCoR --learning_rate 5e-3

# python ./run_experiment.py --model cifar-wideresnet40 --dataset cifar-100 --num_epochs 10 --step 3 --batch_size 32 --balanced --model_suffix worker_raw --uni_name JoCoR --learning_rate 5e-3

# python baseline_code/colearn/main.py --model cifar-wideresnet40 --dataset cifar-100 --num_epochs 10 --step 3 --batch_size 32 --balanced --uni_name JoCoR --learning_rate 5e-3
