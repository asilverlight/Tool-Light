export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=true
module load cuda/12.5.1
module load gcc/13.1.0
python vanilla_sample.py \
    --model_path /path/to/your_model \
    --gpu_use 0.95 \
    --temperature 1 \
    --max_tokens 8000 \
    --max_input_len 32768 \
    --output_path /path/to/your_output.json \
    --batch_size 128 \
    --counts 1300 \
    --data_path /path/to/your_data.json \
    --rollout_counts 10