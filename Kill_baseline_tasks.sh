#!/bin/bash

# 定义要查找并杀掉的进程关键字列表
keywords=("baseline_code" "core_model")

# 循环遍历每个关键字并尝试杀掉相关进程
for keyword in "${keywords[@]}"; do
    echo -e "\e[34m[Info] Searching for processes containing keyword: $keyword\e[0m"
    pids=$(ps aux | grep "$keyword" | grep -v "grep" | awk '{print $2}')

    if [ -z "$pids" ]; then
        echo -e "\e[33m[Warning] No processes found for keyword: $keyword\e[0m"
    else
        for pid in $pids; do
            echo -e "\e[34m[Info] Attempting to kill process with PID: $pid (keyword: $keyword)\e[0m"
            kill -9 "$pid" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo -e "\e[32m[✔] Successfully killed process with PID: $pid\e[0m"
            else
                echo -e "\e[31m[✖] Failed to kill process with PID: $pid\e[0m"
            fi
        done
    fi
done

echo -e "\e[34m[Info] All matching processes have been processed.\e[0m"
