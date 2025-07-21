module load cuda/12.5.1
module load gcc/13.1.0

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /path/to/Qwen2___5-72B-Instruct \
    --served-model-name Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --uvicorn-log-level debug \
    --host 0.0.0.0 \
    --port 28710
