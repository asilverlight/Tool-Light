#!/bin/bash
# Switch to the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "Switched to directory: $SCRIPT_DIR"

# Set Python environment
export PYTHONPATH=$(pwd):$PYTHONPATH

# datasets
data_names=(
    "aime24"
    "aime25"
    "amc23"
    "math500"
    "math"
    "gsm8k"
    "2wiki"
    "bamboogle"
    "musique"
    "hotpotqa"
)

# Reasoning model endpoints
infer_endpoints=(
    "http://localhost:8001/v1"
    "http://localhost:8002/v1"
    "http://localhost:8003/v1"
    "http://localhost:8004/v1"
    "http://localhost:8005/v1"
    "http://localhost:8006/v1"
    "http://localhost:8007/v1"
    "http://localhost:8008/v1"
)  

ENDPOINTS=$(echo "${infer_endpoints[@]}" | tr '\n' ' ')

SAMPLE_TIMEOUT=900  # Timeout for one sample

EXP_NAME="your_model_name"
MODEL_PATH="/path/to/your_model_path"
OUTPUT_PATH="/path/to/your_output_path"   
DATA_PATH="data"           

with_tools=true
if [ "$with_tools" = true ]; then
    PROMPT_TYPE="code_search"          # Prompt type (code_search, search, math, base)
    MAX_PYTHON_TIMES=5                 # Max Python tool invocation times
    MAX_SEARCH_TIMES=5                # Max search tool invocation times
else
    PROMPT_TYPE="base"                 # Prompt type (code_search, search, math, base)
    MAX_PYTHON_TIMES=0                 # Max Python tool invocation times
    MAX_SEARCH_TIMES=0                 # Max search tool invocation times
fi


# VLLM config
echo "Inference endpoints: $ENDPOINTS"
API_KEYS=""                     # API keys list, corresponds to endpoints; empty means default "EMPTY"
DEFAULT_MODEL=$EXP_NAME  # Default model name

# Generation parameters
TEMPERATURE=1                      # Temperature parameter
MAX_TOKENS=4096                     # Max tokens to generate
TOP_P=0.95                          # Top-p truncation
TOP_K=20                           # Top-k truncation
MIN_P=0.0                          # Minimum probability threshold
REPETITION_PENALTY=1.1             # Repetition penalty factor
INCLUDE_STOP_STR=true              # Whether to include stop string in output

# Inference configuration
BATCH_SIZE=8                       # Batch size
MAX_CONCURRENT=50                  # Max concurrent requests
COUNTS=500                        # Number of samples to process

# Tool configurations
CONDA_PATH="/path/to/your/conda/"   # Conda installation path
CONDA_ENV="evaluation_arpo"                                # Conda environment name
PYTHON_MAX_CONCURRENT=32                        # Max concurrent Python executor
BING_API_KEY="<your_bing_api_key>"  # Bing Search API key
BING_ZONE="<your_bing_zone>"                        # Bing search zone
SEARCH_MAX_RESULTS=5                            # Max number of search results
SEARCH_RESULT_LENGTH=1000                        # Max length per search result
BING_REQUESTS_PER_SECOND=32.0                    # Max Bing search requests per second
BING_MAX_RETRIES=3                              # Max Bing search retries
BING_RETRY_DELAY=1.0                            # Bing search retry delay (seconds)
MAX_SEQUENCE_LENGTH=20000                        # Maximum sequence length for summarization

# Simple deep search config
SUMM_MODEL_URLS="http://localhost:8020/v1"
SUMM_MODEL_NAME="Qwen2.5-7B-Instruct"
SUMM_MODEL_PATH="/path/to/Qwen2.5-7B-Instruct"
SEARCH_CACHE_FILE="/path/to/search_cache.db"
URL_CACHE_FILE="/path/to/search_url_cache.db"
USE_LOCAL_SEARCH=false
LOCAL_SEARCH_URL="0.0.0.0:1243"
COMPATIBLE_SEARCH=true
USE_SDS=true

for DATASET_NAME in "${data_names[@]}"; do
    # Build command line arguments
    CMD="python -u infer.py"
    CMD+=" --endpoints $ENDPOINTS"
    CMD+=" --model_path $MODEL_PATH"
    CMD+=" --default_model $DEFAULT_MODEL"

    # If API_KEYS is not empty, add the parameter
    if [ ! -z "$API_KEYS" ]; then
        CMD+=" --api_keys $API_KEYS"
    fi

    # Add generation parameters
    CMD+=" --temperature $TEMPERATURE"
    CMD+=" --max_tokens $MAX_TOKENS"
    CMD+=" --top_p $TOP_P"
    CMD+=" --top_k $TOP_K"
    CMD+=" --min_p $MIN_P"
    CMD+=" --repetition_penalty $REPETITION_PENALTY"
    CMD+=" --include_stop_str_in_output $INCLUDE_STOP_STR"

    # Add inference config parameters
    CMD+=" --max_concurrent_requests $MAX_CONCURRENT"
    CMD+=" --dataset_name $DATASET_NAME"
    CMD+=" --output_path $OUTPUT_PATH"
    CMD+=" --prompt_type $PROMPT_TYPE"
    CMD+=" --counts $COUNTS"
    CMD+=" --max_python_times $MAX_PYTHON_TIMES"
    CMD+=" --max_search_times $MAX_SEARCH_TIMES"
    CMD+=" --sample_timeout $SAMPLE_TIMEOUT"

    # If DATA_PATH is not empty, add the parameter
    if [ ! -z "$DATA_PATH" ]; then
        CMD+=" --data_path $DATA_PATH"
    fi

    # Add tool config parameters
    CMD+=" --conda_path $CONDA_PATH"
    CMD+=" --conda_env $CONDA_ENV"
    CMD+=" --python_max_concurrent $PYTHON_MAX_CONCURRENT"
    CMD+=" --bing_api_key $BING_API_KEY"
    CMD+=" --bing_zone $BING_ZONE"
    CMD+=" --search_max_results $SEARCH_MAX_RESULTS"
    CMD+=" --search_result_length $SEARCH_RESULT_LENGTH"
    CMD+=" --bing_requests_per_second $BING_REQUESTS_PER_SECOND"
    CMD+=" --bing_max_retries $BING_MAX_RETRIES"
    CMD+=" --bing_retry_delay $BING_RETRY_DELAY"

    # Additional parameters for search tool
    CMD+=" --summ_model_urls $SUMM_MODEL_URLS"
    CMD+=" --summ_model_name $SUMM_MODEL_NAME"
    CMD+=" --summ_model_path $SUMM_MODEL_PATH"
    CMD+=" --search_cache_file $SEARCH_CACHE_FILE"
    CMD+=" --url_cache_file $URL_CACHE_FILE"

    if [ "$COMPATIBLE_SEARCH" = true ]; then
        CMD+=" --use_local_search"
        CMD+=" --local_search_url $LOCAL_SEARCH_URL"
        CMD+=" --compatible_search"
    else
        if [ "$USE_LOCAL_SEARCH" = true ]; then
            CMD+=" --use_local_search"
            CMD+=" --local_search_url $LOCAL_SEARCH_URL"
        fi
    fi

    CMD+=" --max_sequence_length $MAX_SEQUENCE_LENGTH"

    if [ "$USE_SDS" = true ]; then
        CMD+=" --use_sds"
    fi

    # Create output directory
    OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
    mkdir -p "$OUTPUT_DIR"
    echo "Created output directory: $OUTPUT_DIR"

    echo $CMD

    # Execute command
    eval $CMD | tee logs/infer.log 2>&1
done