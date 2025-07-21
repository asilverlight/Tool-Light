export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=true
module load cuda/12.5.1
module load gcc/13.1.0

python entropy_guided_sample.py \
    --model_path /path/to/your_model \
    --gpu_use 0.95 \
    --temperature 1 \
    --max_tokens 4096 \
    --max_input_len 32768 \
    --output_path /path/to/your_output.json \
    --batch_size 100 \
    --counts 100 \
    --data_path /path/to/your_data.json \
    --max_rollout_steps 3 \
    --max_rollout_counts 3