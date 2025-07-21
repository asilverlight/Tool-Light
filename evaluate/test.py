import json 
import math
import os
os.environ["TORCH_INDUCTOR_DISABLE_TRITON"] = "1"

from vllm import LLM, SamplingParams

import torch

from tqdm import tqdm
import argparse
import time
from transformers import AutoTokenizer
import requests
from python_executor import PythonExecutor
from web_search.web_search_main import *

# from utils import *
from evaluate.utils import *

class Inference():
    def __init__(self, model, tokenizer, params_config, task, dataset_name, output_path, batch_size=4, counts=100, prompt_type='code_search', data_path=None, use_summarize=False):
        self.model = model
        self.tokenizer = tokenizer
        self.params_config = SamplingParams(**params_config)
        self.task = task
        self.dataset_name = dataset_name
        self.output_path = output_path
        self.batch_size = batch_size
        self.counts = counts
        self.prompt_type = prompt_type
        self.data_path = data_path
        self.use_summarize = use_summarize
        self.prompt_template = ''
        self.max_python_times = 4
        self.max_search_times = 4
        self.questions = []
        self.answers = []
        self.executor = PythonExecutor(get_answer_from_stdout=True)
        self.prompt_template = """You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool and python interpreter tool. \
Given a question, you need to first think about the reasoning process in the mind and then provide the answer. \
During thinking, you can invoke the wikipedia search tool to search and python interpreter tool to calculate the math problem for fact information about specific topics if needed. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, \
and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. \
For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> \
<think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> \
<think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. \
In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format.
"""

    def run(self):
        self.load_datas()
        res = []
        start_time = time.time()
        total_examples = min(len(self.questions), self.counts) if self.counts > 0 else len(self.questions)
        questions = self.questions[:total_examples]
        answers = self.answers[:total_examples]
        num_batches = math.ceil(len(questions) / self.batch_size)
        
        for batch_idx in tqdm(range(num_batches), desc=f"Processing batches"):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(questions))
            batch_samples = questions[start_idx:end_idx]
            golden_answers = answers[start_idx:end_idx]
            
            prompts = []
            for item in batch_samples:
                prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {
                                "role": "system",
                                "content": self.prompt_template
                            },
                            {
                                "role": "user",
                                "content": item
                            }
                        ], tokenize=False, add_generation_prompt=True, add_model_prefix=True
                    )
                )
            
            outputs = []
            generating = list(range(len(prompts))) 
            completed = []  
            concat_prompts_outputs = prompts.copy()  
            python_rounds = [0 for _ in range(len(prompts))]
            search_rounds = [0 for _ in range(len(prompts))]
            
            while generating:
                input_prompts = [concat_prompts_outputs[i] for i in generating]
                self.params_config.stop = ['</python>', '</search>', '</answer>']
                initial_outputs = self.model.generate(
                    input_prompts,
                    self.params_config,
                    use_tqdm=False,
                )
                outputs = [output.outputs[0].text for output in initial_outputs]
                python_indices = []
                search_indices = []
                other_indices = []
                text_generating_indices = []
                
                for i in range(len(outputs)):
                    if outputs[i].strip().endswith('</python>'):
                        if python_rounds[generating[i]] >= self.max_python_times:
                            text_generating_indices.append((generating[i], outputs[i]))
                        else:
                            python_indices.append((generating[i], outputs[i]))
                            python_rounds[generating[i]] += 1
                    elif outputs[i].strip().endswith('</search>'):
                        if search_rounds[generating[i]] >= self.max_search_times:
                            text_generating_indices.append((generating[i], outputs[i]))
                        else:
                            search_indices.append((generating[i], outputs[i]))
                            search_rounds[generating[i]] += 1
                    else:
                        other_indices.append((generating[i], outputs[i]))
                
                if python_indices:
                    print('python begin')
                    python_contents = []
                    for i, content in python_indices:
                        python_contents.append(content)
                        concat_prompts_outputs[i] += content
                    python_contents = [extract_python_content(content) for content in python_contents]
                    for i, (idx, content) in enumerate(python_indices):
                        result, report = self.executor.apply(python_contents[i])
                        # assert False
                        if report == "Done":
                            concat_prompts_outputs[idx] += f'<result>\n{result}\n</result>'
                        else:
                            concat_prompts_outputs[idx] += f'<result>\n{report}\n</result>'
                    print('python end')
                    
                if search_indices:
                    print('search begin')
                    search_contents = []
                    for i, content in search_indices:
                        search_contents.append(
                            content
                        )
                        concat_prompts_outputs[i] += content
                    search_contents = [extract_search_content(content) for content in search_contents]

                    if self.task == 'qa' and self.dataset_name != 'webwalker' and self.dataset_name != 'gpqa' and self.dataset_name != 'hle' and self.dataset_name != 'gaia':
                        search_results = batch_search(search_contents)
                        for i, (idx, content) in enumerate(search_indices):
                            if search_results[i] == 'error':
                                concat_prompts_outputs[idx] += f'<result>\n\n</result>'
                            else:
                                concat_prompts_outputs[idx] += f'<result>\n{search_results[i]}\n</result>'
                    else:
                        for i, (idx, content) in enumerate(search_indices):
                            try:
                                if self.use_summarize:
                                    search_result = deep_search_with_summarize(search_contents[i])
                                else:
                                    search_result = deep_search(search_contents[i])
                                concat_prompts_outputs[idx] += f'<result>\n{search_result}\n</result>'
                            except Exception as e:
                                concat_prompts_outputs[idx] += f'<result>\n\n</result>'
                    print('search end')
                    

                if text_generating_indices:
                    generate_results = []
                    for i, content in text_generating_indices:
                        generate_results.append(
                            concat_prompts_outputs[i] + content
                        )
                        concat_prompts_outputs[i] += content
                    self.params_config.stop = ['</answer>']
                    output_texts = self.model.generate(
                        generate_results,
                        self.params_config,
                        use_tqdm=False,
                    )
                    for i in range(len(output_texts)):
                        text = output_texts[i].outputs[0].text
                        concat_prompts_outputs[text_generating_indices[i][0]] += text
                        completed.append(text_generating_indices[i][0])
                
                if other_indices:
                    for i, content in other_indices:
                        concat_prompts_outputs[i] += content
                        completed.append(i)
                
                generating = [i for i in generating if i not in completed]

            extracted_answers = []
            for i in range(len(concat_prompts_outputs)):
                text = concat_prompts_outputs[i][len(prompts[i]):]
                # Extract answer using the last occurrence of <answer>...</answer>
                # This ensures we get the latest answer in case there are multiple sections
                last_answer_end = text.rfind('</answer>')
                if last_answer_end != -1:
                    # Find the corresponding opening tag before this closing tag
                    temp_text = text[:last_answer_end]
                    last_answer_start = temp_text.rfind('<answer>')
                    if last_answer_start != -1:
                        temp_answer = text[last_answer_start + len('<answer>'):last_answer_end]
                    else:
                        temp_answer = None
                else:
                    temp_answer = None
                if temp_answer:
                    boxed_answer = temp_answer.strip()
                    boxed_answer = last_boxed_only_string(boxed_answer)
                    if boxed_answer and boxed_answer.startswith("\\boxed{") and boxed_answer.endswith("}"):
                        boxed_content = boxed_answer[7:-1]  # Extract content between \\boxed{ and }
                        boxed_answer = boxed_content
                    if not boxed_answer:
                        final_answer = temp_answer
                    else:
                        final_answer = boxed_answer
                else:
                    boxed_answer = text.strip()
                    final_answer = last_boxed_only_string(boxed_answer)
                    if final_answer and final_answer.startswith("\\boxed{") and final_answer.endswith("}"):
                        final_answer = final_answer[7:-1]  # Extract content between \\boxed{ and }
                extracted_answers.append(final_answer)
            
            for i in range(len(batch_samples)):
                res.append(
                    {
                        "Prompt": prompts[i], "Full_output": concat_prompts_outputs[i][len(prompts[i]):], "Output": extracted_answers[i], "answer": golden_answers[i]
                    }
                )
            
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=4, ensure_ascii=False)
            f.close()

    def load_datas(self):
        data_path = self.data_path
        with open(data_path, 'r', encoding='utf-8') as f:
            datas = json.load(f)
            f.close()
        for item in datas:
            self.questions.append(item['question'])
            self.answers.append(item['answer'])


def load_model(config):
        model = LLM(
                    config['model_path'],
                    dtype=config['type'],
                    enforce_eager=True,
                    trust_remote_code=True,
                    max_model_len=config['max_input_len'],
                    gpu_memory_utilization=config['gpu_use'],
                    tensor_parallel_size=config['gpu_num'],
                )
        tokenizer = AutoTokenizer.from_pretrained(config['model_path'], trust_remote_code=True)
        return model, tokenizer

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Torl test")
    argument_parser.add_argument(
        "--model_path",
        type=str,
        default="/models/ToRL",
        help="Model path to use for testing",
    )
    argument_parser.add_argument(
        "--gpu_use",
        type=float,
        default=0.95,
        help="GPU to use for testing",
    )
    argument_parser.add_argument(
        "--temperature",
        type=float,
        default=0,
    )
    argument_parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
    )
    argument_parser.add_argument(
        "--max_input_len",
        type=int,
        default=4096,
    )
    argument_parser.add_argument(
        "--task",
        type=str,
        default='math',
    )
    argument_parser.add_argument(
        "--dataset_name",
        type=str,
        default='none',
    )
    argument_parser.add_argument(
        "--output_path",
        type=str,
        default="/resultsTORL",
        help="Path to the data file",
    )
    argument_parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    argument_parser.add_argument(
        "--prompt_type",
        type=str,
        default='code_search',
    )
    argument_parser.add_argument(
        "--counts",
        type=int,
        default=10000,
    )
    argument_parser.add_argument(
        "--use_debug",
        action='store_true',
    )
    argument_parser.add_argument(
        "--use_rollback",
        action='store_true',
    )
    argument_parser.add_argument(
        "--data_path",
        type=str,
        default=None
    )
    argument_parser.add_argument(
        "--use_summarize",
        action='store_true',
        help="Whether to use summarize tool or not, default is False",
    )
    args = argument_parser.parse_args()

    model_config = {
        'model_path': args.model_path,
        'type': torch.bfloat16,
        'max_input_len': args.max_input_len,
        'gpu_use': args.gpu_use,
        'gpu_num': torch.cuda.device_count(),
        'lora_path': None,
    }
    params_config = {
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'top_p': 0.8,
        'top_k': 20,
        'min_p': 0.0,
        'repetition_penalty': 1.1,
        'n': 1,
        'stop': ['```python'],
        'include_stop_str_in_output': True,
    }
    model, tokenizer = load_model(model_config)
    inference = Inference(
        model=model,
        tokenizer=tokenizer,
        params_config=params_config,
        task=args.task,
        dataset_name=args.dataset_name,
        output_path=args.output_path,
        batch_size=args.batch_size,
        counts=args.counts,
        prompt_type=args.prompt_type,
        data_path=args.data_path,
        use_summarize=args.use_summarize
    )
    inference.run()