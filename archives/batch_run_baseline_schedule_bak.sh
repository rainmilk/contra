#!/bin/bash

# 清空 logs 目录中的所有 .log 文件
rm -f logs/*.log

# 定义任务的脚本，根据显存占用进行分类
light_tasks=(
    # CIFAR-10 任务
    "run_cifar10_cotta.sh"
    "run_cifar10_contra.sh"
    "run_cifar10_coteaching.sh"
    "run_cifar10_coteaching_plus.sh"
    "run_cifar10_plf.sh"
    "run_cifar10_jocor.sh"
    "run_cifar10_replay.sh"
)

# 高显存占用任务
heavy_tasks=(
    "run_cifar100_cotta.sh"
    "run_cifar100_contra.sh"
    "run_cifar100_coteaching.sh"
    "run_cifar100_coteaching_plus.sh"
    "run_cifar100_jocor.sh"
    "run_cifar100_replay.sh"
    "run_pet37_cotta.sh"
    "run_pet37_contra.sh"
    "run_pet37_coteaching.sh"
    "run_pet37_coteaching_plus.sh"
    "run_pet37_jocor.sh"
    "run_pet37_replay.sh"
    "run_pet37_cotta_asy.sh"
    "run_pet37_contra_asy.sh"
    "run_pet37_coteaching_asy.sh"
    "run_pet37_coteaching_plus_asy.sh"
    "run_pet37_jocor_asy.sh"
    "run_pet37_replay_asy.sh"
)

# 中等显存占用任务（这些任务没有明确标记为高显存占用，但可能比轻量任务更耗显存）
medium_tasks=(
    "run_cifar100_plf.sh"
    "run_pet37_plf.sh"
    "run_pet37_plf_asy.sh"
)

# 定义 GPU 使用范围，1-7
gpus=(1 2 3 4 5 6 7)
gpu_count=${#gpus[@]}
gpu_usage=() # 用来追踪每个GPU上的任务
for ((i = 0; i < gpu_count; i++)); do
    gpu_usage[$i]=0 # 初始化每块GPU上的任务数量为0
done

# 创建 logs 目录
mkdir -p logs

# 函数：查找空闲的 GPU
find_free_gpu() {
    min_tasks=${gpu_usage[0]}
    free_gpu=0
    for ((i = 1; i < gpu_count; i++)); do
        if [ ${gpu_usage[$i]} -lt $min_tasks ]; then
            min_tasks=${gpu_usage[$i]}
            free_gpu=$i
        fi
    done
    echo $free_gpu
}

# 运行轻量任务（并行）
for script in "${light_tasks[@]}"; do
    script_name=$(basename "$script" .sh)
    free_gpu=$(find_free_gpu) # 查找空闲 GPU
    gpu_id=${gpus[$free_gpu]}
    gpu_usage[$free_gpu]=$((gpu_usage[$free_gpu] + 1)) # 增加该GPU的任务计数

    log_file="logs/${script_name}.log"

    echo "Running $script on GPU $gpu_id..."
    ./$script $gpu_id >"$log_file" 2>&1 &

    # 确保 GPU 分配不冲突
    sleep 2
done

# 运行中等显存占用任务（并行）
for script in "${medium_tasks[@]}"; do
    script_name=$(basename "$script" .sh)
    free_gpu=$(find_free_gpu) # 查找空闲 GPU
    gpu_id=${gpus[$free_gpu]}
    gpu_usage[$free_gpu]=$((gpu_usage[$free_gpu] + 1)) # 增加该GPU的任务计数

    log_file="logs/${script_name}.log"

    echo "Running $script on GPU $gpu_id..."
    ./$script $gpu_id >"$log_file" 2>&1 &

    sleep 2
done

# 运行高显存占用任务（并行）
for script in "${heavy_tasks[@]}"; do
    script_name=$(basename "$script" .sh)
    free_gpu=$(find_free_gpu) # 查找空闲 GPU
    gpu_id=${gpus[$free_gpu]}
    gpu_usage[$free_gpu]=$((gpu_usage[$free_gpu] + 1)) # 增加该GPU的任务计数

    log_file="logs/${script_name}.log"

    echo "Running $script on GPU $gpu_id..."
    ./$script $gpu_id >"$log_file" 2>&1 &

    # 不使用 wait，允许多个高显存任务并行执行
    sleep 2
done

# 等待所有后台任务完成
wait

echo "所有脚本已完成运行，日志已保存到 logs 目录中。"
