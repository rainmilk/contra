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

'''
the following command has been disused
'''
# python ./core_model/CRUL.py --model wideresnet50 --dataset flower-102 --num_epochs 3 --repair_iter_num 5 --uni_name CRUL --learning_rate 1e-4 --optimizer adam --batch_size 256 --noise_type symmetric --noise_ratio 0

# noise 0.25
python ./core_model/CRUL.py --model wideresnet50 --dataset flower-102 --num_epochs 3 --repair_iter_num 5 --uni_name CRUL --learning_rate 5e-4 --optimizer adam --batch_size 256 --noise_type symmetric --noise_ratio 0.25 --mixup_alpha 0.5
# CUDA_VISIBLE_DEVICES=4 python ./core_model/CRUL.py --model wideresnet50 --dataset flower-102 --num_epochs 3 --repair_iter_num 5 --uni_name CRUL --learning_rate 1e-4 --optimizer adam --batch_size 256 --noise_type symmetric --noise_ratio 0.25 --mixup_alpha 0.5

'''
the following command has been executed in batch_run_our_method_cvpr.sh
'''
# noise 0.5
# python ./core_model/CRUL.py --model wideresnet50 --dataset flower-102 --num_epochs 3 --repair_iter_num 5 --uni_name CRUL --learning_rate 1e-4 --optimizer adam --batch_size 256 --noise_type symmetric --noise_ratio 0.5 --mixup_alpha 0.5
# CUDA_VISIBLE_DEVICES=5 python ./core_model/CRUL.py --model wideresnet50 --dataset flower-102 --num_epochs 3 --repair_iter_num 5 --uni_name CRUL --learning_rate 1e-4 --optimizer adam --batch_size 256 --noise_type symmetric --noise_ratio 0.5 --mixup_alpha 0.5

# noise 0.75
python ./core_model/CRUL.py --model wideresnet50 --dataset flower-102 --num_epochs 3 --repair_iter_num 5 --uni_name CRUL --learning_rate 1e-4 --optimizer adam --batch_size 256 --noise_type symmetric --noise_ratio 0.75 --mixup_alpha 0.5
# CUDA_VISIBLE_DEVICES=6 python ./core_model/CRUL.py --model wideresnet50 --dataset flower-102 --num_epochs 3 --repair_iter_num 5 --uni_name CRUL --learning_rate 1e-4 --optimizer adam --batch_size 256 --noise_type symmetric --noise_ratio 0.75 --mixup_alpha 0.5
