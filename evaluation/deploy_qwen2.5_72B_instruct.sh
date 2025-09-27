#!/bin/bash
# source /path/to/your/conda/bin/activate
# conda activate vllm_arpo


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve /path/to/Qwen2.5-72B-Instruct \
  --served-model-name Qwen2.5-72B-Instruct \
  --max-model-len 32768 \
  --tensor_parallel_size 8 \
  --gpu-memory-utilization 0.95 \
  --port 8001 \
  --max-logprobs 100