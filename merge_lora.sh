module load cuda/12.5.1
module load gcc/13.1.0

python merge_lora.py \
    --base_model /path/to/your_model_path \
    --lora_model /path/to/your_lora_path \
    --output_path /path/to/your_model_path_merge_lora