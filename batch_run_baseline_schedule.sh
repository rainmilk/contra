#!/bin/bash

# 清空 logs 目录中的所有 .log 文件
rm -f logs/*.log

# 定义任务的脚本列表
tasks=(
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
gpu_usage=() # 用来追踪每个GPU上的任务状态
for ((i = 0; i < gpu_count; i++)); do
    gpu_usage[$i]=0 # 初始化每块GPU的任务数量为0
done

# 创建 logs 目录
mkdir -p logs

# 函数：查找空闲的 GPU（当前没有高负荷任务）
find_free_gpu() {
    while true; do
        for ((i = 0; i < gpu_count; i++)); do
            if [ ${gpu_usage[$i]} -eq 0 ]; then
                echo $i
                return
            fi
        done
        # 等待 GPU 变空闲
        sleep 5
    done
}

# 函数：为指定的 GPU 运行任务
run_task_on_gpu() {
    local script=$1
    local gpu_id=$2
    script_name=$(basename "$script" .sh)

    log_file="logs/${script_name}.log"

    echo "Running $script on GPU $gpu_id..."
    ./$script $gpu_id >"$log_file" 2>&1 &
    local pid=$!

    # 等待任务完成，并释放该GPU
    wait $pid
    gpu_usage[$gpu_id]=0
}

# 逐个分配任务
for script in "${tasks[@]}"; do
    free_gpu=$(find_free_gpu) # 查找空闲 GPU
    gpu_id=${gpus[$free_gpu]}
    gpu_usage[$free_gpu]=1 # 标记该 GPU 正在运行任务

    # 在后台执行任务
    run_task_on_gpu $script $gpu_id &
done

# 等待所有后台任务完成
wait

echo "所有脚本已完成运行，日志已保存到 logs 目录中。"
