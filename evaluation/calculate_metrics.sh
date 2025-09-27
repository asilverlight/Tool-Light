export PYTHONPATH=/path/to/Tool-Light/:$PYTHONPATH

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

methods=(
    "efficiency"
    "necessity"
)

for data_name in "${data_names[@]}"; do
    for method in "${methods[@]}"; do
        echo "Calculating metrics for $data_name with method $method"
        python calculate_metrics.py \
            --output_path /path/to/${method}/${data_name}_${method}.json \
            --other_paths /path/to/search_o1/Search_o1_${data_name}_result.metrics.json,/path/to/search_r1/Search_R1_${data_name}_result.metrics.json,/path/to/recall/ReCall_${data_name}_result.metrics.json,/path/to/dotamath/DOTAMATH_${data_name}_result.metrics.json,/path/to/torl/ToRL_${data_name}_result.metrics.json,/path/to/prompt_base/Prompt_Base_${data_name}_result.metrics.json,/path/to/retool/ReTool_${data_name}_result.metrics.json,/path/to/tool_star_7b_sft/Tool-Star-SFT_${data_name}_result.metrics.json,/path/to/tool-light/Tool-Light_${data_name}_result.metrics.json \
            --exp_type $method \
            --dataset $data_name 
    done
done