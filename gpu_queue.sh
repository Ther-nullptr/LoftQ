#!/bin/bash

# 设置要轮询的GPU ID
GPU_ID=1

# 设置轮询间隔（秒）
INTERVAL=60

# 设置要运行的程序
YOUR_PROGRAM="bash run_wikitext_2.sh"

# 获取GPU显存使用情况函数
function get_gpu_memory_usage() {
    nvidia-smi --id=$1 --query-gpu=memory.used --format=csv,noheader,nounits
}

# 主循环
while true; do
    # 获取GPU显存使用情况
    memory_used=$(get_gpu_memory_usage $GPU_ID)

    # 如果显存为0，则允许程序运行
    if [ $memory_used -lt 100 ]; then
        echo "GPU memory is free. Starting your program..."
        $YOUR_PROGRAM
        break
    fi

    # 等待指定的间隔后再次检查
    sleep $INTERVAL
done
