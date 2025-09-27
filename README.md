<h1 align="center"> 💡Toward Effective Tool-Integrated Reasoning via
Self-Evolved Preference Learning</a></h1>

<!-- <div align="center"> 

[![Model](https://img.shields.io/badge/Model-Hugging%20Face-blue?logo=huggingface)](https://huggingface.co/collections/dongguanting/arpo-688229ff8a6143fe5b4ad8ae)
[![Model](https://img.shields.io/badge/Model-ModelScope-blue?logo=ModelScope)](https://modelscope.cn/models/dongguanting/arpo-688229ff8a6143fe5b4ad8ae)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg)](https://opensource.org/licenses/MIT) 
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
</div> -->

<div align="center">
  <a href="https://huggingface.co/zhangboguodong/Tool-Light-Qwen2.5-7B-it">
    <img src="https://img.shields.io/badge/Model-Hugging%20Face-blue?logo=huggingface" alt="Hugging Face Models">
  </a>
  <a href="https://modelscope.cn/models/zhangboguodong/Tool_Light_Qwen2.5_7B_it">
    <img src="https://img.shields.io/badge/Model-ModelScope-blue?logo=" alt="ModelScope Models">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/LICENSE-MIT-green.svg" alt="License">
  </a>
  <a href="https://www.python.org/downloads/release/python-390/">
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  </a>
</div>


## 😃 Overview


**Tool-Light** is a framework focused on enabling models to efficiently complete TIR tasks. Tool-Light innovatively introduces the **Entropy-Guided Sampling Strategy** to construct the training set. Besides, it trains the model through the **Self-Evolved DPO Pipeline**. This design empowers the model to gradually acquire the ability to call tools efficiently and accurately. Results on two types of reasoning tasks demonstrate superior performance compared to traditional methods.

<!-- <p align="center">
<img width="100%" alt="image" src="https://github.com/asilverlight/Tool-Light/blob/main/figs/algorithm.png" />
</p> -->
![image](figs/algorithm.png)

## 😋 Quick Start for Data Construction
### 1. Environment Setup

In this step, we should first operate SFT on Qwen2.5-7B-Instruct model. Please first set up the environment for [Llama Factory](https://github.com/hiyouga/LLaMA-Factory).

```bash
git clone https://github.com/asilverlight/Tool-Light/
cd Tool-Light/LLaMA-Factory-main

conda create -n sft python=3.10
conda activate sft

pip install -r requirements.txt
```

### 2. Conduct SFT on Qwen2.5-7B-Instruct

1. Download your SFT dataset from [🤗Tool-Star-SFT-54K](https://huggingface.co/datasets/dongguanting/Tool-Star-SFT-54K) and place it in `LLaMA-Factory-main/data/final_sft_edition9.json`. Define the dataset in `dataset_info.json`. By the way, [Tool-Star](https://github.com/RUC-NLPIR/Tool-Star) is also a wonderful work:)

2. In `LLaMA-Factory-main/examples/train_full/llama_factory.sh`, execute the following codes:

```bash
module load cuda/12.1.1
cd /path/to/LLaMA-Factory-main
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train /path/to/LLaMA-Factory-main/examples/train_full/sft.yaml
```
### 3. Inference Environment Setup
First, configure the required environment.
```bash
conda create -n inference python==3.10
conda activate inference
pip install -r ./Tool-Light/evaluation/requirement.txt
```
### 4. Use SFT Model to Select Source Datas
Use the SFT model to directly perform inference on `LLaMA-Factory-main/data/final_sft_edition9.json`, and screen out the data sources for DPO training. You can execute the following code:
```bash
bash entropy_guided_sample/run.sh
```
### 5. Use Two Strategies to Sample Datas
Based on the data sources you've screened out, use the SFT model for sampling.

Execute the following code in `entropy_guided_sample/vanilla_sample.sh` for Vanilla Sampling:
```bash 
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
    --counts 4000 \
    --data_path /path/to/your_data.json \
    --rollout_counts 10
```
Execute the following code in `entropy_guided_sample/entropy_guided_sample.sh` for Entropy-Guided Sampling:
```bash 
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
    --counts 4000 \
    --data_path /path/to/your_data.json \
    --max_rollout_steps 3 \
    --max_rollout_counts 3
```
![image](figs/tree_sampling.png)
### 6. Construct Positive-Negative Examples According to Criteria
For Pre-Aligned DPO and Self-Evolved On-Policy DPO parts, we design different criteria for screening positive-negative examples. You can refer to the description in the paper, and then construct the training set for the two types of sampled data.

## 🥰 Self-Evolved DPO Training
### 1. Environment Setup
This part is the same as **Environment Setup** in **Quick Start for Data Construction**.
### 2. Conduct DPO Training
1. Define your constructed DPO dataset in `dataset_info.json`.

2. In `LLaMA-Factory-main/examples/train_full/llama_factory.sh`, execute the following codes:

```bash
module load cuda/12.1.1
conda activate sft
cd /path/to/LLaMA-Factory-main
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train /path/to/LLaMA-Factory-main/examples/train_full/step1_dpo.yaml
```
3. Enter the `inference` environment. Then, use the DPO model to sample again from the same 4000 data sources. After that, screen the positive-negative examples according to the criteria of the Self-Evolved On-Policy DPO Loop phase.

4. Define this phase's training data, and execute the following codes in `LLaMA-Factory-main/examples/train_full/llama_factory.sh`:
```bash
module load cuda/12.1.1
conda activate sft
cd /path/to/LLaMA-Factory-main
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train /path/to/LLaMA-Factory-main/examples/train_full/step2_dpo.yaml
```
### 3. Evaluate the Performance of Trained Model

1. Enter the `inference` environment.

2. Deploy the retriever for performing search tasks on Wikipedia-based datasets. We provide a Wikipedia retriever service implemented using FlashRAG and FastAPI. Before starting the retriever serving, you need to download the [pre-indexed Wikipedia](https://github.com/RUC-NLPIR/FlashRAG?tab=readme-ov-file#index), [Wikipedia corpus, and corresponding retriever models](https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/original_docs/reproduce_experiment.md#preliminary). The corpuses used can be found [here](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/retrieval-corpus), and Index construction method can be found [here](https://github.com/RUC-NLPIR/FlashRAG/tree/main?tab=readme-ov-file#rocket-quick-start).

More details can be found in the [FlashRAG documentation](https://github.com/RUC-NLPIR/FlashRAG/tree/main?tab=readme-ov-file#rocket-quick-start).

To start the retriever serving, first fill in `evaluate/retriever/serving_config.yaml` with the correct paths to the retrieval model, index, and corpus, as well as available GPU IDs. Then, run the following command to start the retriever serving:

```bash
python host_wiki.py \
    --config serving_config.yaml \
    --num_retriever {num_retriever} \
    --port {port}
```
3. Deploy judging model. You can execute the following code:
```bash
bash evaluate/deploy_qwen2.5_72B_instruct.sh
```
4. Execute code in `Tool-Light/evaluation/infer_local_sds.sh` to evaluate the performance of the model. 

First, run the following code in `Tool-Light/evaluation/infer_local_sds.sh` for inference:
```bash
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
```
And then, you can test the metrics of the model.

First, run the code in `Tool-Light/evaluation/deploy_qwen2.5_72B_instruct.sh` to deploy the judging model.

Then, run the code in `Tool-Light/evaluation/evaluate_all_datas.sh` to evaluate the performance of the model. Here, we evaluate the **F1 score** and the **LLM-as-Judge** metric.

More details for evaluation can be found in [ARPO](https://github.com/RUC-NLPIR/ARPO).

5. For the measurement of the **Efficiency** and **Necessity** metrics, please run `evaluate/calculate_metrics.sh`:
```bash
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
            --dataset $data_name \
            --model_path /path/to/Qwen2.5-7B-Instruct
    done
done
```
 Note that for the measurement of these two metrics, you need to run `Tool-Light/evaluation/evaluate_all_datas.sh` before. For the measurement of **Necessity**, you need to obtain the F1 scores and the LLM-as-Judge metrics for all baselines in advance.
