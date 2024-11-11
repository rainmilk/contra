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

# Full model
python ./core_model/CRUL.py --uni_name CRUL_full --model wideresnet50 --dataset pet-37 --learning_rate 2e-4 --optimizer adam --batch_size 256 --noise_type asymmetric --noise_ratio 0.5 --repair_iter_num 5 --num_epochs 5 --ul_epochs 3 --agree_epochs 3 --mixup_alpha 0.75 --ls_gamma=0.2 --temperature=0.8

# w/o Unlearning
python ./core_model/CRUL.py --uni_name CRUL_wo_ul --model wideresnet50 --dataset pet-37 --learning_rate 2e-4 --optimizer adam --batch_size 256 --noise_type asymmetric --noise_ratio 0.5 --repair_iter_num 5 --num_epochs 5 --ul_epochs 0 --agree_epochs 3 --mixup_alpha 0.75 --ls_gamma=0.2 --temperature=0.8

# w/o High-confidence agreement label smoothing
python ./core_model/CRUL.py --uni_name CRUL_wo_ls --model wideresnet50 --dataset pet-37 --learning_rate 2e-4 --optimizer adam --batch_size 256 --noise_type asymmetric --noise_ratio 0.5 --repair_iter_num 5 --num_epochs 5 --ul_epochs 3 --agree_epochs 0 --mixup_alpha 0.75 --ls_gamma=0.2 --temperature=0.8

# w/o Unlearning and High-confidence agreement label smoothing
python ./core_model/CRUL.py --uni_name CRUL_wo_ulls --model wideresnet50 --dataset pet-37 --learning_rate 2e-4 --optimizer adam --batch_size 256 --noise_type asymmetric --noise_ratio 0.5 --repair_iter_num 5 --num_epochs 5 --ul_epochs 0 --agree_epochs 0 --mixup_alpha 0.75 --ls_gamma=0.2 --temperature=0.8

# w/o mixup
python ./core_model/CRUL.py --uni_name CRUL_wo_mixup --model wideresnet50 --dataset pet-37 --learning_rate 2e-4 --optimizer adam --batch_size 256 --noise_type asymmetric --noise_ratio 0.5 --repair_iter_num 5 --num_epochs 5 --ul_epochs 3 --agree_epochs 3 --mixup_alpha 0.001 --ls_gamma=0.2 --temperature=0.8