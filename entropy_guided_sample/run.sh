export CUDA_VISIBLE_DEVICES=0,2
export TOKENIZERS_PARALLELISM=true

python run.py \
    --data_path /path/to/final_sft_edition9.json \
    --output_path /path/to/your_output_datas.json \
    --model_name qwen2.5-7b-instruct \
    --gpu_use 0.95 \
    --max_tokens 2048 \
    --max_input_len 16384 \
    --temperature 1 