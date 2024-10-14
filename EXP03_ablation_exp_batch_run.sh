#!/bin/bash

# 备份 logs 目录中的所有 .log 文件
backup_dir="logs_backup/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"
mv logs/*.log "$backup_dir" 2>/dev/null


# 定义任务的脚本列表

tasks=(
    ############################################
    # 分类任务 - PET37 - 对称噪声 - Resnet50
    # task: image classification
    # dataset: pet37
    # noise type: sy
    # noise ratio: 0.2
    # forget ratio: 0.5/0.3/0.1
    # method: raw/coteaching/contra
    # 消融01: 仅有 repair
    # Add 10-10 已经生成
    # "run_pet37_raw_repair_only.sh"
    # "run_pet37_conteaching_repair_only.sh"
    # "run_pet37_contra_repair_only.sh"
    ######################################
    # 消融02: 仅有 tta
    # "run_pet37_raw_tta_only.sh"
    # "run_pet37_conteaching_tta_only.sh"
    "run_pet37_contra_tta_only.sh"
    ############################################
    # 消融03: 有 repair + 有 tta, done

    ############################################
    # 检索任务 - PET37 - 非对称噪声 - WideResnet50
    # task: image classification
    # dataset: pet37
    # noise type: sy
    # noise ratio: 0.2
    # forget ratio: 0.5/0.3/0.1
    # method: raw/coteaching/contra
    # 消融01: 仅有 repair
    # "run_pet37_raw_asy_repair_only.sh"
    # "run_pet37_conteaching_asy_repair_only.sh"
    # "run_pet37_contra_asy_repair_only.sh"
    ######################################
    # 消融02: 仅有 tta
    # "run_pet37_raw_asy_tta_only.sh"
    # "run_pet37_conteaching_asy_tta_only.sh"
    "run_pet37_contra_asy_tta_only.sh"
    ############################################
    # 消融03: 有 repair + 有 tta, done
)

# 检查所有任务脚本是否存在
check_only=false
if [ "$1" == "--check-only" ]; then
    check_only=true
fi

all_present=true
for task in "${tasks[@]}"; do
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
gpus=(3 4)
gpu_count=${#gpus[@]}

# Create logs directory
mkdir -p logs

# Initialize an associative array to keep track of GPU usage
declare -A gpu_pids

# Function to check if a PID is still running
is_running() {
    kill -0 "$1" 2>/dev/null
}

# Loop over the tasks
for task in "${tasks[@]}"; do
    # for task in "${tasks_rehs[@]}"; do
    # for task in "${tasks_all[@]}"; do
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

# Wait for all tasks to finish
for pid in "${gpu_pids[@]}"; do
    if [ -n "$pid" ] && is_running "$pid"; then
        wait "$pid"
    fi
done

echo "All tasks have been completed. Logs are saved in the logs directory."
