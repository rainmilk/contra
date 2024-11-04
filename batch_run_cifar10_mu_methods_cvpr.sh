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

sh run_cifar10_mu_raw_cvpr.sh $CUDA_VISIBLE_DEVICES        # raw
sh run_cifar10_mu_ga_cvpr.sh $CUDA_VISIBLE_DEVICES         # GA
sh run_cifar10_mu_gal1_cvpr.sh $CUDA_VISIBLE_DEVICES       # GA_l1
sh run_cifar10_mu_ft_cvpr.sh $CUDA_VISIBLE_DEVICES         # FT
sh run_cifar10_mu_ftl1_cvpr.sh $CUDA_VISIBLE_DEVICES       # FT_l1
sh run_cifar10_mu_fisher_new_cvpr.sh $CUDA_VISIBLE_DEVICES # fisher_new
sh run_cifar10_mu_wfisher_cvpr.sh $CUDA_VISIBLE_DEVICES    # wfisher
sh run_cifar10_mu_ftprune_cvpr.sh $CUDA_VISIBLE_DEVICES    # FT_prune
sh run_cifar10_mu_ftprunebi_cvpr.sh $CUDA_VISIBLE_DEVICES  # FT_prune_bi
sh run_cifar10_mu_retrain_cvpr.sh $CUDA_VISIBLE_DEVICES    # retrain
sh run_cifar10_mu_retrainls_cvpr.sh $CUDA_VISIBLE_DEVICES  # retrain_ls
sh run_cifar10_mu_retrainsam_cvpr.sh $CUDA_VISIBLE_DEVICES # retrain_sam
