#!/bin/bash

use_qwen3=true

# Activate the Conda environment
source /path/to/your/conda/bin/activate
conda activate arpo

# Move to the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "cd $SCRIPT_DIR"

# Create log directory
mkdir -p logs

MODEL_PATH="/path/to/your_model_path"
MODEL_NAME="your_model_name"


echo "Starting Instance on GPU 0"
CUDA_VISIBLE_DEVICES=0 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.4 \
    --enable-chunked-prefill \
    --enforce-eager \
    --port 8001 \
    --no-enable-prefix-caching > logs/your_model_name_0.log 2>&1 &
INSTANCE3_PID=$!
echo "Instance deployed on port 8001 using GPU 0"

echo "Starting Instance on GPU 1"
CUDA_VISIBLE_DEVICES=1 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.4 \
    --enable-chunked-prefill \
    --enforce-eager \
    --port 8002 \
    --no-enable-prefix-caching > logs/your_model_name_1.log 2>&1 &
INSTANCE3_PID=$!
echo "Instance deployed on port 8002 using GPU 1"

echo "Starting Instance on GPU 2"
CUDA_VISIBLE_DEVICES=2 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.4 \
    --enable-chunked-prefill \
    --enforce-eager \
    --port 8003 \
    --no-enable-prefix-caching > logs/your_model_name_2.log 2>&1 &
INSTANCE3_PID=$!
echo "Instance deployed on port 8003 using GPU 2"

echo "Starting Instance on GPU 3"
CUDA_VISIBLE_DEVICES=3 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.4 \
    --enable-chunked-prefill \
    --enforce-eager \
    --port 8004 \
    --no-enable-prefix-caching > logs/your_model_name_3.log 2>&1 &
INSTANCE3_PID=$!
echo "Instance deployed on port 8004 using GPU 3"


# Display all running model services
echo "---------------------------------------"
echo "All deployed model instances:"
ps aux | grep "vllm serve" | grep -v grep
echo "---------------------------------------"

# Handle cleanup on termination
trap "kill $INSTANCE3_PID" SIGTERM
wait $INSTANCE3_PID
