#!/bin/bash

# 清空 logs 目录中的所有 .log 文件
rm -f logs/*.log

# 定义任务的脚本列表

tasks=(
    ######################################
    # 分类任务 - PET37 - 对称噪声 - Resnet50
    # task: image classification
    # dataset: pet37
    # noise type: sy
    # noise ratio: 0.5/0.3/0.1
    # method: raw/coteaching/contra
    "run_pet37_raw_nr_10_pec.sh"
    "run_pet37_raw_nr_30_pec.sh"
    "run_pet37_raw_nr_50_pec.sh"
    "run_pet37_contra_nr_10_pec.sh"
    "run_pet37_contra_nr_30_pec.sh"
    "run_pet37_contra_nr_50_pec.sh"
    "run_pet37_coteaching_nr_10_pec.sh"
    "run_pet37_coteaching_nr_30_pec.sh"
    "run_pet37_coteaching_nr_50_pec.sh"
    # "run_pet37_coteaching_plus.sh"
    # "run_pet37_cotta.sh"
    # "run_pet37_jocor.sh"
    # "run_pet37_plf.sh"
    # "run_pet37_rehearsal.sh"
    ############################################
    # 检索任务 - PET37 - 非对称噪声 - WideResnet50
    # task: image search
    # dataset: pet37
    # noise type: asy
    # noise ratio: 0.5/0.3/0.1
    # method: raw/coteaching/contra
    "run_pet37_raw_asy_nr_10_pec.sh"
    "run_pet37_raw_asy_nr_30_pec.sh"
    "run_pet37_raw_asy_nr_50_pec.sh"
    "run_pet37_contra_asy_nr_10_pec.sh"
    "run_pet37_contra_asy_nr_30_pec.sh"
    "run_pet37_contra_asy_nr_50_pec.sh"
    "run_pet37_coteaching_asy_nr_10_pec.sh"
    "run_pet37_coteaching_asy_nr_30_pec.sh"
    "run_pet37_coteaching_asy_nr_50_pec.sh"
    # "run_pet37_coteaching_plus_asy.sh"
    # "run_pet37_cotta_asy.sh"
    # "run_pet37_jocor_asy.sh"
    # "run_pet37_plf_asy.sh"
    # "run_pet37_rehearsal_asy.sh"
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
# gpus=(1 2 3 4 5 6 7)

gpus=(0 1 2 3 4 5 6 7)

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
