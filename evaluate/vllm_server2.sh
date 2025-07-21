module load cuda/12.5.1
module load gcc/13.1.0
CUDA_VISIBLE_DEVICES=2 vllm serve /path/to/Qwen2.5-7B-Instruct \
    --served-model-name Qwen2.5-7B-Instruct \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --uvicorn-log-level debug \
    --host 0.0.0.0 \
    --port 28712