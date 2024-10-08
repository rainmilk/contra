#!/bin/bash

# 定义要运行的脚本文件数组
scripts=(
    # 分类任务 - CIFAR-10 - 对称噪声 - Resnet18
    "run_cifar10_contra.sh"
    "run_cifar10_coteaching_plus.sh"
    "run_cifar10_coteaching.sh"
    "run_cifar10_cotta.sh"
    "run_cifar10_jocor.sh"
    "run_cifar10_plf.sh"
    "run_cifar10_replay.sh"
    # 分类任务 - PET37 - 对称噪声 - Resnet50
    "run_pet37_contra.sh"
    "run_pet37_coteaching.sh"
    "run_pet37_coteaching_plus.sh"
    "run_pet37_cotta.sh"
    "run_pet37_jocor.sh"
    "run_pet37_plf.sh"
    "run_pet37_replay.sh"
    # 检索任务 - CIFAR-100 - 非对称噪声 - WideResnet40
    "run_cifar100_contra.sh"
    "run_cifar100_coteaching_plus.sh"
    "run_cifar100_coteaching.sh"
    "run_cifar100_cotta.sh"
    "run_cifar100_jocor.sh"
    "run_cifar100_plf.sh"
    "run_cifar100_replay.sh"
    # 检索任务 - PET37 - 非对称噪声 - WideResnet50
    "run_pet37_contra_asy.sh"
    "run_pet37_coteaching_asy.sh"
    "run_pet37_coteaching_plus_asy.sh"
    "run_pet37_cotta_asy.sh"
    "run_pet37_jocor_asy.sh"
    "run_pet37_plf_asy.sh"
    "run_pet37_replay_asy.sh"
)

# 定义 GPU 使用范围，1-7
gpus=(1 2 3 4 5 6 7)
gpu_count=${#gpus[@]}
gpu_index=0  # 用于循环分配 GPU

# 创建 logs 目录
mkdir -p logs

# 循环运行脚本
for script in "${scripts[@]}"; do
    # 获取脚本的基础文件名 (不包含路径)
    script_name=$(basename "$script" .sh)

    # 计算要使用的 GPU ID，按顺序循环分配 GPU
    gpu_id=${gpus[$gpu_index]}
    gpu_index=$(( (gpu_index + 1) % gpu_count ))

    # 创建脚本的日志文件路径
    log_file="logs/${script_name}.log"

    # 运行脚本并重定向输出到日志文件
    echo "Running $script on GPU $gpu_id..."
    echo $script_name
    echo $gpu_id
    echo $log_file

    ./$script $gpu_id >"$log_file" 2>&1 &
    sleep 2

done

# 等待所有后台任务完成
wait

echo "所有脚本已完成运行，日志已保存到 logs 目录中。"
