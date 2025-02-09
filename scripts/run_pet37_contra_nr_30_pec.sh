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

python ./run_experiment.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --step 0 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --noise_type symmetric  --uni_name contra --noise_ratio 0.3

python ./core_model/train_teacher.py --model wideresnet50 --dataset pet-37 --num_epochs 5 --step 0 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --noise_type symmetric  --uni_name contra --noise_ratio 0.3

python ./run_experiment.py --model wideresnet50 --dataset pet-37 --num_epochs 5 --step 1 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --model_suffix worker_raw --noise_type symmetric  --uni_name contra --noise_ratio 0.3

python ./core_model/core.py --model wideresnet50 --dataset pet-37 --num_epochs 1 --step 1 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --noise_type symmetric  --uni_name contra --noise_ratio 0.3

python ./run_experiment.py --model wideresnet50 --dataset pet-37 --num_epochs 5 --step 2 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --model_suffix worker_raw --noise_type symmetric  --uni_name contra --noise_ratio 0.3

python ./core_model/core.py --model wideresnet50 --dataset pet-37 --num_epochs 1 --step 2 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --noise_type symmetric  --uni_name contra --noise_ratio 0.3

python ./run_experiment.py --model wideresnet50 --dataset pet-37 --num_epochs 5 --step 3 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --model_suffix worker_raw --noise_type symmetric  --uni_name contra --noise_ratio 0.3

python ./core_model/core.py --model wideresnet50 --dataset pet-37 --num_epochs 1 --step 3 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --noise_type symmetric  --uni_name contra --noise_ratio 0.3
