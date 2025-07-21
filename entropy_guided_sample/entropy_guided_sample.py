import json 
import math

import os
os.environ["TORCH_INDUCTOR_DISABLE_TRITON"] = "1"

from vllm import LLM, SamplingParams

import torch

from tqdm import tqdm
import argparse
import re
import time
from transformers import AutoTokenizer
from python_executor import PythonExecutor


from entropy_guided_sample.utils import *

class Inference():
    def __init__(self, model, tokenizer, params_config, dataset_name, output_path, batch_size=4, counts=100, data_path=None, temperature=0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.params_config = SamplingParams(**params_config, truncate_prompt_tokens=10000)
        self.dataset_name = dataset_name
        self.output_path = output_path
        self.batch_size = batch_size
        self.counts = counts
        self.data_path = data_path
        self.temperature = temperature
        self.max_python_times = 4
        self.max_search_times = 4
        self.max_rollout_steps = 3  # max steps to rollout
        self.max_rollout_counts = 3  # max rollout counts for each step
        self.questions = []
        self.answers = []
        self.ids = []
        self.sft_label_paths = []
        self.executor = PythonExecutor(get_answer_from_stdout=True)
        self.prompt_template = """You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool and python interpreter tool. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, you can invoke the wikipedia search tool to search and python interpreter tool to calculate the math problem for fact information about specific topics if needed. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> <think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> <think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format."""

    def calculate_sequence_entropy(self, output, text, n_tokens_list=[10, 20, 30, 40, 50]):
        tokens = self.tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")[0]
        all_entropy_values = []
        for n in n_tokens_list:
            if n == 0 or len(tokens) <= n:
                tokens_subset = tokens
            else:
                tokens_subset = tokens[:n]

            temp_text = ''

            current_entropies = []
            for i in range(len(tokens_subset)):
                try:
                    logprob_info = output.outputs[0].logprobs[i]
                except:
                    logprob_info = output.outputs[0].logprobs[-1]

                token_list = list(logprob_info.values())

                temp_text += token_list[0].decoded_token

                logprob_list = [token.logprob for token in token_list]
                entropy_list = [-logprob * math.exp(logprob) for logprob in logprob_list]
                entropy_val = sum(entropy_list)
                current_entropies.append(entropy_val)
            entropy = sum(current_entropies) / len(current_entropies) if current_entropies else 0.0
            all_entropy_values.append((entropy, n, temp_text))
        return all_entropy_values


    def run(self):
        self.load_datas()
        res = []
        start_time = time.time()
        total_examples = min(len(self.questions), self.counts) if self.counts > 0 else len(self.questions)
        questions = self.questions[:total_examples]
        answers = self.answers[:total_examples]
        ids = self.ids[:total_examples]
        sft_label_paths = self.sft_label_paths[:total_examples]
        
        self.params_config.logprobs = 50  
        
        main_chains, main_chains_entropy_info, main_chains_step_entropies = self.batch_generate_main_chains(questions)

        print(f'Main chains generated: {len(main_chains)}')
        
        all_branch_requests = []
        for idx, (main_chain, entropy_info) in enumerate(zip(main_chains, main_chains_entropy_info)):
            rollout_positions = self.select_rollout_positions(entropy_info)
            
            for position, entropy_value, tokens_count, step_idx in rollout_positions:
                for temp_idx in range(self.max_rollout_counts):
                    all_branch_requests.append({
                        "question_idx": idx,
                        "main_chain": main_chain,
                        "position": position,
                        "step_idx": step_idx,
                    })
        self.params_config.temperature = self.temperature  
        all_branch_chains, branch_chains_step_entropies = self.batch_generate_branch_chains(all_branch_requests)
        
        for idx in range(total_examples):
            main_chain = main_chains[idx]
            main_chain_entropy = main_chains_step_entropies[idx]
            
            branch_chains = [chain for chain in all_branch_chains if chain["question_idx"] == idx]
            batch_entropies = [branch_chains_step_entropies[i] for i, chain in enumerate(all_branch_chains) if chain["question_idx"] == idx]
            
            all_chains = [main_chain] + branch_chains
            all_chains_entropy = [main_chain_entropy] + batch_entropies

            for i in range(1, len(all_chains_entropy)):
                if all_chains[i]['step_idx'] > 0:
                    all_chains_entropy[i] = all_chains_entropy[0][:all_chains[i]['step_idx']] + all_chains_entropy[i]
            
            all_outputs = []
            for chain in all_chains:
                output = self.extract_answer(chain["full_output"])
                all_outputs.append(output)
            
            res.append({
                "Prompt": main_chain["prompt"],
                "Main_chain": main_chain["full_output"],
                "All_chains": all_chains,
                "Output": all_outputs,
                "answer": answers[idx],
                "id": ids[idx],
                "sft_label_path": sft_label_paths[idx],
                "entropies": all_chains_entropy,
            })
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=4, ensure_ascii=False)
            f.close()

    def batch_generate_main_chains(self, questions):
        total_examples = len(questions)
        num_batches = math.ceil(total_examples / self.batch_size)
        
        all_main_chains = []
        all_entropy_info = []
        all_step_entropies = []
        
        for batch_idx in tqdm(range(num_batches), desc="Generating main chains"):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, total_examples)
            batch_questions = questions[start_idx:end_idx]
            
            prompts = []
            for question in batch_questions:
                prompt = self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": self.prompt_template
                        },
                        {
                            "role": "user",
                            "content": question
                        }
                    ], tokenize=False, add_generation_prompt=True, add_model_prefix=True
                )
                prompts.append(prompt)
            
            concat_prompts_outputs = prompts.copy()
            python_rounds = [0] * len(batch_questions)
            search_rounds = [0] * len(batch_questions)
            
            entropy_info = [[] for _ in range(len(batch_questions))]
            step_idx = [0] * len(batch_questions) 

            step_entropies = [[] for _ in range(len(batch_questions))]
            
            generating = list(range(len(batch_questions)))
            completed = []
            
            while generating:
                current_prompts = [concat_prompts_outputs[i] for i in generating]
                
                self.params_config.stop = ['</python>', '</search>', '</answer>']

                initial_outputs = self.model.generate(
                    current_prompts,
                    self.params_config,
                    use_tqdm=False,
                )
                
                outputs = [output.outputs[0].text for output in initial_outputs]

                for i, (initial_output, output) in enumerate(zip(initial_outputs, outputs)):
                    gen_idx = generating[i]
                    entropy_value = self.calculate_sequence_entropy(initial_output, output, [0])[0][0]
                    step_entropies[gen_idx].append(entropy_value)

                entropy_values = [
                    self.calculate_sequence_entropy(initial_output, output) for initial_output, output in zip(initial_outputs, outputs)
                ] 
                
                python_indices = []
                search_indices = []
                other_indices = [] 
                text_generating_indices = [] # continue generating
                
                for i, output in enumerate(outputs):
                    gen_idx = generating[i]
                    
                    max_entropy, max_tokens, max_output_text = max(entropy_values[i], key=lambda x: x[0])

                    entropy_info[gen_idx].append({
                        "step": step_idx[gen_idx],
                        "output": output,
                        "position": len(concat_prompts_outputs[gen_idx]) + len(max_output_text) - len(prompts[gen_idx]), 
                        "entropy_values": entropy_values[i],
                        "max_entropy": max_entropy,
                        "max_tokens": max_tokens
                    })
                    step_idx[gen_idx] += 1
                    
                    if output.strip().endswith('</python>'):
                        if python_rounds[gen_idx] < self.max_python_times:
                            python_indices.append((i, gen_idx))
                        else:
                            text_generating_indices.append((i, gen_idx))
                    elif output.strip().endswith('</search>'):
                        if search_rounds[gen_idx] < self.max_search_times:
                            search_indices.append((i, gen_idx))
                        else:
                            text_generating_indices.append((i, gen_idx))
                    else:
                        other_indices.append((i, gen_idx))

                if python_indices:
                    print('python begin')
                    for i, gen_idx in python_indices:
                        concat_prompts_outputs[gen_idx] += outputs[i]
                        python_content = extract_python_content(outputs[i])
                        python_rounds[gen_idx] += 1
                        
                        result, report = self.executor.apply(python_content)
                        if report == "Done":
                            concat_prompts_outputs[gen_idx] += f'<result>\n{result}\n</result>'
                        else:
                            concat_prompts_outputs[gen_idx] += f'<result>\n{report}\n</result>'
                    print('python end')
                
                if search_indices:
                    search_contents = []
                    search_gen_indices = []
                    
                    for i, gen_idx in search_indices:
                        concat_prompts_outputs[gen_idx] += outputs[i]
                        search_content = extract_search_content(outputs[i])
                        search_contents.append(search_content)
                        search_gen_indices.append(gen_idx)
                        search_rounds[gen_idx] += 1
                    
                    print('search begin')
                    search_results = batch_search(search_contents, top_n=3)
                    print('search end')
                    for j, gen_idx in enumerate(search_gen_indices):
                        if search_results[j] == 'error':
                            concat_prompts_outputs[gen_idx] += f'<result>\n\n</result>'
                        else:
                            concat_prompts_outputs[gen_idx] += f'<result>\n{search_results[j]}\n</result>'
                
                if text_generating_indices:
                    continue_prompts = []
                    continue_gen_indices = []
                    
                    for i, gen_idx in text_generating_indices:
                        continue_prompts.append(concat_prompts_outputs[gen_idx] + outputs[i])
                        concat_prompts_outputs[gen_idx] += outputs[i]
                        continue_gen_indices.append(gen_idx)
                    
                    self.params_config.stop = ['</answer>']

                    continue_outputs = self.model.generate(
                        continue_prompts,
                        self.params_config,
                        use_tqdm=False,
                    )
                    
                    for j, gen_idx in enumerate(continue_gen_indices):
                        output = continue_outputs[j].outputs[0].text
                        concat_prompts_outputs[gen_idx] += output
                        
                        entropy_value = self.calculate_sequence_entropy(continue_outputs[j], output, [0])[0][0]
                        entropy_values_continue = self.calculate_sequence_entropy(continue_outputs[j], output)
                        step_entropies[gen_idx].append(entropy_value)
                        max_entropy, max_tokens, max_output_text = max(entropy_values_continue, key=lambda x: x[0])

                        entropy_info[gen_idx].append({
                            "step": step_idx[gen_idx],
                            "output": output,
                            "position": len(concat_prompts_outputs[gen_idx]) - len(output) + len(max_output_text) - len(prompts[gen_idx]),
                            "entropy_values": entropy_values_continue,
                            "max_entropy": max_entropy,
                            "max_tokens": max_tokens
                        })
                        step_idx[gen_idx] += 1
                        
                        completed.append(gen_idx)
                if other_indices:
                    for i, gen_idx in other_indices:
                        concat_prompts_outputs[gen_idx] += outputs[i]
                        completed.append(gen_idx)
                
                generating = [i for i in generating if i not in completed]
            
            for i, prompt in enumerate(prompts):
                idx = start_idx + i
                main_chains = {
                    "prompt": prompt,
                    "full_output": concat_prompts_outputs[i][len(prompt):],
                    "python_rounds": python_rounds[i],
                    "search_rounds": search_rounds[i],
                    "question_idx": idx
                }
                all_main_chains.append(main_chains)
                all_entropy_info.append(entropy_info[i])
                all_step_entropies.append(step_entropies[i])
        
        return all_main_chains, all_entropy_info, all_step_entropies

    def select_rollout_positions(self, entropy_info):
        sorted_steps = sorted(entropy_info, key=lambda x: x["max_entropy"], reverse=True)
        
        selected_positions = []

        for i in range(min(self.max_rollout_steps, len(sorted_steps))):
            step = sorted_steps[i]
            position = step["position"]
            selected_positions.append((position, step["max_entropy"], step["max_tokens"], step["step"]))
        
        return selected_positions

    def batch_generate_branch_chains(self, branch_requests):
        all_branch_chains = []
        all_step_entropies = []
        
        num_batches = math.ceil(len(branch_requests) / self.batch_size)
        
        for batch_idx in tqdm(range(num_batches), desc="Generating branch chains"):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(branch_requests))
            batch_requests = branch_requests[start_idx:end_idx]
            
            prefixes = []
            python_rounds = []
            search_rounds = []
            
            for i, request in enumerate(batch_requests):
                main_chain = request["main_chain"]
                position = request["position"]
                
                prefix = main_chain["prompt"] + main_chain["full_output"][:position]
                
                python_matches = re.findall(r'<python>.*?</python>', main_chain["full_output"][:position], re.DOTALL)
                search_matches = re.findall(r'<search>.*?</search>', main_chain["full_output"][:position], re.DOTALL)
                used_python_rounds = len(python_matches)
                used_search_rounds = len(search_matches)
                
                prefixes.append(prefix)
                python_rounds.append(used_python_rounds)
                search_rounds.append(used_search_rounds)
            generating = list(range(len(batch_requests)))
            completed = []
            concat_prompts_outputs = prefixes.copy()

            step_entropies = [[] for _ in range(len(batch_requests))]
            
            while generating:
                input_prompts = [concat_prompts_outputs[i] for i in generating]
                self.params_config.stop = ['</python>', '</search>', '</answer>']

                initial_outputs = self.model.generate(
                    input_prompts,
                    self.params_config,
                    use_tqdm=False,
                )

                outputs = [output.outputs[0].text for output in initial_outputs]

                for i, (initial_output, output) in enumerate(zip(initial_outputs, outputs)):
                    gen_idx = generating[i]
                    entropy_value = self.calculate_sequence_entropy(initial_output, output, [0])[0][0]
                    step_entropies[gen_idx].append(entropy_value)

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
                        if report == "Done":
                            concat_prompts_outputs[idx] += f'<result>\n{result}\n</result>'
                        else:
                            concat_prompts_outputs[idx] += f'<result>\n{report}\n</result>'
                    print('python end')
                    
                if search_indices:
                    print('search begin')
                    search_contents = []
                    for i, content in search_indices:
                        search_contents.append(content)
                        concat_prompts_outputs[i] += content
                    search_results = batch_search(search_contents, top_n=3)
                    for i, (idx, content) in enumerate(search_indices):
                        if search_results[i] == 'error':
                            concat_prompts_outputs[idx] += f'<result>\n\n</result>'
                        else:
                            concat_prompts_outputs[idx] += f'<result>\n{search_results[i]}\n</result>'
                    print('search end')
                    
                if text_generating_indices:
                    generate_results = []
                    for i, content in text_generating_indices:
                        generate_results.append(concat_prompts_outputs[i] + content)
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

                        idx = text_generating_indices[i][0]
                        entropy_value = self.calculate_sequence_entropy(output_texts[i], text, [0])[0][0]
                        step_entropies[idx].append(entropy_value)

                        completed.append(text_generating_indices[i][0])
                
                if other_indices:
                    for i, content in other_indices:
                        concat_prompts_outputs[i] += content
                        completed.append(i)
                
                generating = [i for i in generating if i not in completed]
            
            for i, request in enumerate(batch_requests):
                full_output = concat_prompts_outputs[i][len(request["main_chain"]["prompt"]):]
                branch_chain = {
                    "prompt": request["main_chain"]["prompt"],
                    "full_output": full_output,
                    "step_idx": request["step_idx"],
                    "python_rounds": python_rounds[i],
                    "search_rounds": search_rounds[i],
                    "question_idx": request["question_idx"]
                }
                all_branch_chains.append(branch_chain)
                all_step_entropies.append(step_entropies[i])
        
        return all_branch_chains, all_step_entropies

    def extract_answer(self, text):
        # Extract answer using the last occurrence of <answer>...</answer>
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
                
        return final_answer

    def load_datas(self):
        data_path = self.data_path
        with open(data_path, 'r', encoding='utf-8') as f:
            datas = json.load(f)
            f.close()
        for item in datas:
            if 'question' in item:
                self.questions.append(item['question'])
            else:
                prompt = item['Prompt']
                user_start = prompt.rfind('<|im_start|>user\n')
                user_start += len('<|im_start|>user\n')
                assistant_start = prompt.find('<|im_end|>\n<|im_start|>assistant\n', user_start)
                question = prompt[user_start:assistant_start].strip()
                self.questions.append(question)
            if 'golden answer' in item:
                self.answers.append(item['golden answer'])
            else:
                self.answers.append(item['answer'])
            self.ids.append(item['id'])
            if 'output' in item:
                self.sft_label_paths.append(item['output'])
            else:
                self.sft_label_paths.append(item['sft_label_path'])


def load_model(config):
        model = LLM(
                    config['model_path'],
                    dtype=config['type'],
                    enforce_eager=True,
                    trust_remote_code=True,
                    max_model_len=config['max_input_len'],
                    gpu_memory_utilization=config['gpu_use'],
                    tensor_parallel_size=config['gpu_num'],
                    max_logprobs=50,  
                )
        tokenizer = AutoTokenizer.from_pretrained(config['model_path'], trust_remote_code=True)
        return model, tokenizer

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Torl test with entropy-based rollouts")
    argument_parser.add_argument(
        "--model_path",
        type=str,
        default="/path/models/ToRL",
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
        "--dataset_name",
        type=str,
        default='none',
    )
    argument_parser.add_argument(
        "--output_path",
        type=str,
        default="/results/TORL",
        help="Path to the data file",
    )
    argument_parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    argument_parser.add_argument(
        "--counts",
        type=int,
        default=10000,
    )
    argument_parser.add_argument(
        "--data_path",
        type=str,
        default=None
    )
    argument_parser.add_argument(
        "--max_rollout_steps",
        type=int,
        default=3,
        help="Maximum number of rollout steps",
    )
    argument_parser.add_argument(
        "--max_rollout_counts",
        type=int,
        default=4,
        help="Maximum number of rollout branches per position",
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
        dataset_name=args.dataset_name,
        output_path=args.output_path,
        batch_size=args.batch_size,
        counts=args.counts,
        data_path=args.data_path,
        temperature=args.temperature,
    )
    inference.max_rollout_steps = args.max_rollout_steps
    inference.max_rollout_counts = args.max_rollout_counts
    inference.run()
