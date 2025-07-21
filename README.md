<h1 align="center"> 💡Toward Effective Tool-Integrated Reasoning via
Self-Evolved Preference Learning</a></h1>


## 😃 Overview


**Tool-Light** is a framework focused on enabling models to efficiently complete TIR tasks. Tool-Light innovatively introduces the **Entropy-Guided Sampling Strategy** to construct the training set. Besides, it trains the model through the **Self-Evolved DPO Pipeline**. This design empowers the model to gradually acquire the ability to call tools efficiently and accurately. Results on two types of reasoning tasks demonstrate superior performance compared to traditional methods.

## 😋 Quick Start for Data Construction
### 1. Environment Setup

In this step, we should first operate SFT on Qwen2.5-7B-Instruct model. Please first set up the environment for [Llama Factory](https://github.com/hiyouga/LLaMA-Factory).

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[metrics]"
```

### 2. Conduct SFT on Qwen2.5-7B-Instruct

1. Download your SFT dataset from [🤗Tool-Star-SFT-54K](https://huggingface.co/datasets/dongguanting/Tool-Star-SFT-54K) and place it in `LLaMA-Factory-main/data/final_sft_edition9.json`. Define the dataset in `dataset_info.json`.

2. In `LLaMA-Factory-main/examples/train_full/llama_factory.sh`, execute the following codes:

```bash
module load cuda/12.1.1
cd /path/to/LLaMA-Factory-main
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train /path/to/LLaMA-Factory-main/examples/train_full/sft.yaml
```
### 3. Inference Environment Setup
First, configure the required environment.
```bash
#create env
conda create -n toollight python==3.10
conda activate toollight

# install torch & flash-atten
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation

# This is our env freeze file. You can install it as a supplement or use it for checking.
pip install -r ./Tool-Light/requirements.txt
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
cd /path/to/LLaMA-Factory-main
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train /path/to/LLaMA-Factory-main/examples/train_full/step1_dpo.yaml
```
3. Enter the `Tool-Light` environment. Then, use the DPO model to sample again from the same 4000 data sources. After that, screen the positive-negative examples according to the criteria of the Self-Evolved On-Policy DPO Loop phase.

4. Define this phase's training data, and execute the following codes in `LLaMA-Factory-main/examples/train_full/llama_factory.sh`:
```bash
module load cuda/12.1.1
cd /path/to/LLaMA-Factory-main
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train /path/to/LLaMA-Factory-main/examples/train_full/step2_dpo.yaml
```
### 3. Evaluate the Performance of Trained Model
1. Enter the `Tool-Light` environment.
2. Deploy the retriever for performing search tasks on Wikipedia-based datasets. We provide a Wikipedia retriever service implemented using FlashRAG and FastAPI. Before starting the retriever serving, you need to download the [pre-indexed Wikipedia](https://github.com/RUC-NLPIR/FlashRAG?tab=readme-ov-file#index), [Wikipedia corpus, and corresponding retriever models](https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/original_docs/reproduce_experiment.md#preliminary). The corpuses used can be found [here](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/retrieval-corpus), and Index construction method can be found [here](https://github.com/RUC-NLPIR/FlashRAG/tree/main?tab=readme-ov-file#rocket-quick-start).

More details can be found in the [FlashRAG documentation](https://github.com/RUC-NLPIR/FlashRAG/tree/main?tab=readme-ov-file#rocket-quick-start).

To start the retriever serving, first fill in `evaluate/retriever/serving_config.yaml` with the correct paths to the retrieval model, index, and corpus, as well as available GPU IDs. Then, run the following command to start the retriever serving:

```bash
python host_wiki.py \
    --config serving_config.yaml \
    --num_retriever {num_retriever} \
    --port {port}
```
3. Deploy judging model. You can execute the following code in `evaluate/vllm_server.sh`:
```bash
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
```
4. Execute code in `evaluate/test.sh` to evaluate the performance of the model. Here, we evaluate the **F1 score** and the **LLM-as-Judge** metric:
```bash
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
        python evaluate.py \
            --output_path /path/to/${output_path}/${exp_type}_${data_name}_result.json \
            --task ${task} \
            --dataset_name ${data_name} \
            --use_llm \
            --extract_answer
    done
done
```
5. For the measurement of the **Efficiency** and **Effectiveness** metrics, please run `evaluate/calculate_metrics.sh`:
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
    "accuracy"
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
 Note that for the measurement of these two metrics, you need to run `evaluate/test.sh` before. For the measurement of **Effectiveness**, you need to obtain the F1 scores and the LLM-as-Judge metrics for all baselines in advance.