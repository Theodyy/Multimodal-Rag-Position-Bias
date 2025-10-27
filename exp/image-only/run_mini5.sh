#!/bin/bash

base_command="python chart-mini.py --dataset_name chart --output_dir ../positionbias/attention/singleattention/result/mini_5_output_dir"

# 定义输出文件名
output_file="script_output_mini5.log"

# 循环执行命令
for i in {1..10}
do
    command="$base_command$i"
    # 记录日期和时间戳
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] Running command: $command" | tee -a "$output_file"
    # 输出分隔符
    echo "------------------------" | tee -a "$output_file"
    # 执行命令并记录输出
    $command 2>&1 | tee -a "$output_file"
    # 检查命令执行状态
    if [ $? -ne 0 ]; then
        echo "[$timestamp] Error occurred while running command: $command" | tee -a "$output_file"
        # 若需要在出错时终止脚本，取消注释下面这行
        # exit 1
    fi
    # 输出分隔符
    echo "========================" | tee -a "$output_file"
done    