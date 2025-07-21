export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=true
module load cuda/12.5.1
module load gcc/13.1.0


data_names=(
    "hotpotqa"
    "2wiki"
    "bamboogle"
    "musique"
    "aime24"
    "aime25"
    "gsm8k"
    "math"
    "math500"
    "amc23"
)

tasks=("qa" "qa" "qa" "qa" "math" "math" "math" "math" "math" "math") # 
exp_types=("DPO") 
models=("/path/to/tool-light")  
output_paths=("dpo") 
for i in "${!data_names[@]}"
do
    data_name="${data_names[$i]}"
    task="${tasks[$i]}"
    for j in "${!exp_types[@]}"
    do
        exp_type="${exp_types[$j]}"
        output_path="${output_paths[$j]}"
        python test.py \
            --model_path "${models[$j]}" \
            --dataset_name "${data_name}" \
            --task "${task}" \
            --gpu_use 0.95 \
            --max_tokens 8192 \
            --max_input_len 32768 \
            --temperature 1 \
            --output_path /path/to/${output_path}/${exp_type}_${data_name}_result.json \
            --counts 500 \
            --batch_size 50 \
            --data_path datas/${data_name}/test.json
        python /home/u2024001021/agentic_search/for_quick_hand/evaluate_dgt.py \
            --output_path /fs/archive/share/START/evaluation_dgt/${output_path}/${exp_type}_${data_name}_result.json \
            --task ${task} \
            --dataset_name ${data_name} \
            --use_llm \
            --extract_answer
    done
done
