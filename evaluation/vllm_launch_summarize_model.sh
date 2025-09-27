#!/bin/bash

# Activate the Conda environment
source /path/to/your/conda/bin/activate
conda activate arpo

# Switch to the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "Switched to directory: $SCRIPT_DIR"

# Create log directory
mkdir -p logs

# Model path - all instances use the same model
MODEL_NAME="Qwen2.5-7B-Instruct"
MODEL_PATH="/path/to/Qwen2.5-7B-Instruct"

# Launch Instance 1 - using GPU 0
echo "Starting Instance 1 on GPU 5"
CUDA_VISIBLE_DEVICES=5 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 1 \
    --gpu-memory-utilization 0.95 \
    --port 8020 > logs/summarize_model1.log 2>&1 &
INSTANCE1_PID=$!
echo "Instance 1 deployed on port 8020 using GPU 5"


# Display all running model services
echo "---------------------------------------"
echo "All deployed model instances:"
ps aux | grep "vllm serve" | grep -v grep
echo "---------------------------------------"

# Gracefully terminate both instances on SIGTERM
trap "kill $INSTANCE1_PID" SIGTERM
wait $INSTANCE1_PID
