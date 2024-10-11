#!/bin/bash

# 备份 logs 目录中的所有 .log 文件
backup_dir="logs_backup/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"
mv logs/*.log "$backup_dir" 2>/dev/null

# 定义任务的脚本列表
tasks=(
    # 分类任务 - CIFAR-10 - 对称噪声 20% - 遗忘 0.5/0.3/0.1 - Resnet18
    # "run_cifar10_contra.sh"
    "run_cifar10_coteaching_plus.sh"
    "run_cifar10_coteaching.sh"
    "run_cifar10_cotta.sh"
    "run_cifar10_jocor.sh"
    "run_cifar10_plf.sh"
    "run_cifar10_raw.sh"
    "run_cifar10_rehearsal.sh"
    # 分类任务 - PET37 - 对称噪声 20% - 遗忘 0.5/0.3/0.1 - Resnet50
    # "run_pet37_contra.sh"
    "run_pet37_coteaching.sh"
    "run_pet37_coteaching_plus.sh"
    "run_pet37_cotta.sh"
    "run_pet37_jocor.sh"
    "run_pet37_plf.sh"
    "run_pet37_raw.sh"
    "run_pet37_rehearsal.sh"
    # 检索任务 - CIFAR-100 - 非对称噪声 20% - 遗忘 0.5/0.3/0.1 - WideResnet40
    # "run_cifar100_contra.sh"
    "run_cifar100_coteaching_plus.sh"
    "run_cifar100_coteaching.sh"
    "run_cifar100_cotta.sh"
    "run_cifar100_jocor.sh"
    "run_cifar100_plf.sh"
    "run_cifar100_raw.sh"
    "run_cifar100_rehearsal.sh"
    # 检索任务 - PET37 - 非对称噪声 20% - 遗忘 0.5/0.3/0.1 - WideResnet50
    # "run_pet37_contra_asy.sh"
    "run_pet37_coteaching_asy.sh"
    "run_pet37_coteaching_plus_asy.sh"
    "run_pet37_cotta_asy.sh"
    "run_pet37_jocor_asy.sh"
    "run_pet37_plf_asy.sh"
    "run_pet37_raw_asy.sh"
    "run_pet37_rehearsal_asy.sh"
)

tasks_contra=(
    "run_cifar10_contra.sh"
    "run_pet37_contra.sh"
    "run_cifar100_contra.sh"
    "run_pet37_contra_asy.sh"
)

# tasks_contra=(
#     "run_cifar10_contra.sh"
#     # "run_pet37_contra.sh"
#     # "run_cifar100_contra.sh"
#     # "run_pet37_contra_asy.sh"
# )

# 检查所有任务脚本是否存在
check_only=false
if [ "$1" == "--check-only" ]; then
    check_only=true
fi

all_present=true
for task in "${tasks_contra[@]}" "${tasks[@]}"; do
    if [ -f "$task" ]; then
        echo -e "\e[32m[✔] Checked: $task - File exists\e[0m"
    else
        echo -e "\e[31m[✖] Error: $task - File not found\e[0m"
        all_present=false
    fi
done

if [ "$all_present" == false ]; then
    echo -e "\e[31mOne or more scripts are missing. Exiting.\e[0m"
    exit 1
fi

if [ "$check_only" == true ]; then
    echo -e "\e[34mAll scripts are present. Exiting as per --check-only flag.\e[0m"
    exit 0
fi

# Define the GPUs to use
# gpus=(1 2 3 4 5 6 7)
gpus=(2 3 4 5 6 7)
gpu_count=${#gpus[@]}

# Create logs directory
mkdir -p logs

# Initialize an associative array to keep track of GPU usage
declare -A gpu_pids

# Function to check if a PID is still running
is_running() {
    kill -0 "$1" 2>/dev/null
}

# Function to run tasks on available GPUs
run_tasks() {
    local tasks_list=("$@")
    for task in "${tasks_list[@]}"; do
        # Find a free GPU
        while true; do
            for gpu_id in "${gpus[@]}"; do
                pid=${gpu_pids[$gpu_id]}
                if [ -z "$pid" ] || ! is_running "$pid"; then
                    # GPU is free; assign the task
                    script_name=$(basename "$task" .sh)
                    log_file="logs/${script_name}.log"
                    echo "Running $task on GPU $gpu_id..."
                    ./$task "$gpu_id" >"$log_file" 2>&1 &
                    gpu_pids[$gpu_id]=$!
                    sleep 2 # Brief pause before assigning the next task
                    break 2 # Exit both loops
                fi
            done
            sleep 2 # Wait before checking for free GPUs again
        done
    done
}

# Run tasks_contra first, then tasks
run_tasks "${tasks_contra[@]}"
run_tasks "${tasks[@]}"

# Wait for all tasks to finish
for pid in "${gpu_pids[@]}"; do
    if [ -n "$pid" ] && is_running "$pid"; then
        wait "$pid"
    fi
done

echo "All tasks have been completed. Logs are saved in the logs directory."
