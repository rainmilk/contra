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

# [24-10-10, Add by sunzekun]
# 由于run_experiment.py的逻辑发生变化，现在要先不指定--uni_name，生成一次存储在/pretrain/step_0的模型，才能正常执行后续的操作
# 如果通过其它指令，在对应组下有/pretrain/step_0 的模型，可以注释掉第一行。
# python ./run_experiment.py --model wideresnet50 --dataset pet-37 --num_epochs 20 --step 0 --learning_rate 0.0001 --optimizer adam --batch_size 64 --balanced --model_suffix worker_restore

python ./run_experiment.py --model wideresnet50 --dataset pet-37 --num_epochs 20 --step 0 --learning_rate 0.0001 --optimizer adam --batch_size 64 --uni_name raw --balanced --model_suffix worker_restore

python ./run_experiment.py --model wideresnet50 --dataset pet-37 --num_epochs 20 --step 1 --learning_rate 0.0001 --optimizer adam --batch_size 64 --uni_name raw --balanced --model_suffix worker_restore

python ./run_experiment.py --model wideresnet50 --dataset pet-37 --num_epochs 20 --step 2 --learning_rate 0.0001 --optimizer adam --batch_size 64 --uni_name raw --balanced --model_suffix worker_restore

python ./run_experiment.py --model wideresnet50 --dataset pet-37 --num_epochs 20 --step 3 --learning_rate 0.0001 --optimizer adam --batch_size 64 --uni_name raw --balanced --model_suffix worker_restore
