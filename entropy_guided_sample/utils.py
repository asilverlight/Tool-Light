from tqdm import tqdm
import argparse
import re
from transformers import AutoTokenizer
from vllm import LLM
import time
import datetime
import requests
from typing import List, Dict, Optional, final, Union
from matplotlib import pyplot as plt
from openai import OpenAI
import os
import numpy as np
import asyncio
from openai import AsyncOpenAI
from equivalence import is_equiv
rng = np.random.default_rng(4396)

model2path = {
        'qwen2-7b-instruct': '/path/to/Qwen2-7B-Instruct',
        'glm-4-9b-chat': "/path/to/models/ZhipuAI/glm-4-9b-chat",
        'mistral-7b-instruct': '/path/to/Mistral-7B-Instruct-v0.3',
        'llama2-7b-chat': '/path/to/llama-2-7b-chat-hf',
        'llama3-8b-instruct': '/path/to/Meta-Llama-3-8B-Instruct',
        'qwen2.5-7b-instruct': '/path/to/Qwen2.5-7B-Instruct',
        'llama3.1-8b-instruct': '/path/to/Llama-3.1-8B-Instruct',
        'llama3.2-1b-instruct': '/path/to/LLM-Research/Llama-3___2-1B-Instruct',
        'llama3.2-3b-instruct': '/path/to/LLM-Research/Llama-3___2-3B-Instruct',
        'llama3.3-70b-instruct': '/path/to/llama3.3-70B-instruct',
        'qwen2.5-1.5b-instruct': '/path/to/Qwen/Qwen2___5-1___5B-Instruct',
        'qwen2.5-0.5b-instruct': '/path/to/Qwen/Qwen2___5-0___5B-Instruct',
        
        'deepseek-llm-7b-chat': '/path/to/deepseek-ai/deepseek-llm-7b-chat',
        'baichuan2-7b-chat': '/path/to/baichuan-inc/Baichuan2-7B-Chat',
        'qwen2.5-3b-instruct': '/path/to/Qwen/Qwen2___5-3B-Instruct',
        'openelm-3b-instruct': '/path/to/LLM-Research/OpenELM-3B-Instruct',
        'qwen2.5-3b': '/path/to/Qwen/Qwen2___5-3B',
        'gemma-7b-it': '/path/to/LLM-Research/gemma-7b-it',
        'gemma-2-2b-it': '/path/to/LLM-Research/gemma-2-2b-it',
        'gemma-2b-it': '/path/to/AI-ModelScope/gemma-2b-it',
        'qwen2.5-14b-instruct': '/path/to/Qwen/Qwen2___5-14B-Instruct',
        'qwen2.5-32b-instruct': '/path/to/Qwen/Qwen2___5-32B-Instruct',
        'qwen2.5-72b-instruct': '/path/to/qwen2.5-72B-Instruct',
        'qwen3-8b': '/path/to/START/models/Qwen/Qwen3-8B',
        'e5': "/path/to/e5-base-v2",
    }

async def llm_evaluate_equivalence_single(
    client: AsyncOpenAI,
    question: str,
    labeled_answer: str,
    pred_answer: str,
    model_name: str,
    semaphore: asyncio.Semaphore,
    retry_limit: int = 3,
) -> bool:
    """Evaluate a single pair of answers using LLM"""
#     else:
    prompt = f"""You are an evaluation assistant. Please determine if the model output is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {labeled_answer}

Model Output (Last few lines): {pred_answer}

Did the model give an answer equivalent to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
"""

    for attempt in range(retry_limit):
        try:
            async with semaphore:
                chat_response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                reason_answer = chat_response.choices[0].message.content.strip()
                try:
                    start = reason_answer.index("<judgment>") + len("<judgment>")
                    end = reason_answer.index("</judgment>")
                    response_text = reason_answer[start:end].strip()
                except:
                    response_text = reason_answer.strip()
                llm_judge = is_equiv(pred_answer, labeled_answer) or \
                    "correct" in response_text.lower() and \
                    not ("incorrect" in response_text.lower() or \
                         "wrong" in response_text.lower() or \
                         "not correct" in response_text.lower())
                return llm_judge
        except Exception as e:
            if attempt == retry_limit - 1:
                # import pdb; pdb.set_trace()
                print(f"Error in LLM evaluation: {e}")
                print(f"-------------------pred_answer: {pred_answer}----------------------")
                print(f"-------------------labeled_answer: {labeled_answer}----------------------")
                return is_equiv(pred_answer, labeled_answer)
            await asyncio.sleep(1 * (attempt + 1))
    
    return is_equiv(pred_answer, labeled_answer)

async def llm_evaluate_equivalence_batch(
    questions: List[str],
    labeled_answers: List[str], 
    pred_answers: List[str],
    api_base_url: str = None,
    model_name: str = None,
    api_key: str = "empty",
    concurrent_limit: int = 50,
) -> List[bool]:
    """
    Evaluate multiple answer pairs concurrently using LLM
    """
    if api_base_url is None:
        api_base_url = "http://0.0.0.0:28710/v1"
    if model_name is None:
        model_name = "Qwen2.5-72B-Instruct"

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=api_base_url,
    )

    semaphore = asyncio.Semaphore(concurrent_limit)
    
    tasks = [
        llm_evaluate_equivalence_single(
            client=client,
            question=q,
            labeled_answer=l,
            pred_answer=p,
            model_name=model_name,
            semaphore=semaphore,
        )
        for q, l, p in zip(questions, labeled_answers, pred_answers)
    ]

    with tqdm(total=len(tasks), desc="LLM Evaluation") as pbar:
        async def track_progress(task):
            result = await task
            pbar.update(1)
            return result
            
        tracked_tasks = [track_progress(task) for task in tasks]
        results = await asyncio.gather(*tracked_tasks)
    
    return results

def search(query: str):
    if query == '':
        return 'invalid query'

    url = f'http://your_api/search'
    data = {'query': query, 'top_n': 5}
    # 发送POST请求
    response = requests.post(url, json=data)
    # 初始化检索文本
    retrieval_text = ''
    # 处理返回的JSON数据
    for line in response.json():
        # 将每条检索结果添加到retrieval_text中
        retrieval_text += f"{line['contents']}\n\n"
    # 去除首尾空白字符
    retrieval_text = retrieval_text.strip()
    return retrieval_text

def batch_search(query: Union[str, List[str]], top_n=4) -> List[str]:
    if len(query) == 0:
        return 'invalid query'

    url = f'http://0.0.0.0:1243/batch_search'
    if isinstance(query, str):
        query = [query]
    data = {'query': query, 'top_n': top_n}
    response = requests.post(url, json=data)
    
    result_list = []
    for item in response.json():
        curr_result = ''
        for line in item:
            curr_result += f"{line['contents']}\n\n"
        result_list.append(curr_result.strip())
    
    return result_list

def extract_search_content(text: str) -> str:
    try:
        # 定义搜索内容的起始和结束标签
        start_tag = '<search>'
        end_tag = '</search>'
        # 确保文本以结束标签结尾
        assert text.strip().endswith(end_tag)
        # 找到结束标签的位置
        end_pos = text.rindex(end_tag)
        # 找到起始标签的位置
        start_pos = text.rindex(start_tag, 0, end_pos)
        # 提取并返回标签之间的内容
        return text[start_pos + len(start_tag):end_pos].strip()
    except ValueError:
        # 如果提取失败，返回空字符串
        return ""

def validate_template_format(text: str) -> tuple[bool, str]:
    """
    检查文本是否是有效的QA模板格式
    返回: (是否有效, 错误信息)
    """
    # 检查 <think></think> 标签是否成对出现
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> 标签不成对"
    
    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "缺少 <think> 或 </think> 标签"
    
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> 或 </answer> 标签出现次数不为1"        
    
    # 检查 search/result 标签的顺序
    current_pos = 0
    while True:
        search_pos = text.find('<search>', current_pos)
        if search_pos == -1:
            break
            
        result_pos = text.find('<result>', search_pos)
        search_end_pos = text.find('</search>', search_pos)
        result_end_pos = text.find('</result>', result_pos)
        
        if -1 in (result_pos, search_end_pos, result_end_pos):
            return False, "search/result 标签不完整"
            
        if not (search_pos < search_end_pos < result_pos < result_end_pos):
            return False, "search/result 标签嵌套顺序错误"
            
        current_pos = result_end_pos
    
    # 检查答案中是否包含 \boxed{} 格式
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> 必须在 </answer> 之前"
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "答案缺少 \\boxed{} 格式"
    
    return True, "格式正确"

def extract_answer(text: str):
    """从文本中提取答案部分"""
    text = text.strip()

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    
    return match.group(1)

def remove_boxed(s):
    """移除 \boxed{} 或 \boxed 格式，只保留内容"""
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]

def extract_python_content(text: str) -> str:
    try:
        # 定义Python内容的起始和结束标签
        start_tag = '<python>'
        end_tag = '</python>'
        # 确保文本以结束标签结尾
        assert text.strip().endswith(end_tag)
        # 找到结束标签的位置
        end_pos = text.rindex(end_tag)
        # 找到起始标签的位置
        start_pos = text.rindex(start_tag, 0, end_pos)
        # 提取并返回标签之间的内容
        return text[start_pos + len(start_tag):end_pos].strip()
    except ValueError:
        # 如果提取失败，返回空字符串
        return ""


def last_boxed_only_string(string):
    """
    提取字符串中最后一个 \boxed{} 或 \fbox{} 格式的内容
    如果找不到则返回 None
    """
    try:
        idx = string.rfind("\\boxed")
    except:
        import pdb; pdb.set_trace()
    # if "\\boxed " in string:
    #     return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = string[idx:]
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def compute_score_with_format(solution_str, ground_truth) -> float:
    """
    计算答案的得分
    检查格式是否正确，答案是否与标准答案匹配
    返回: (得分, 原因说明)
    """
    solution_str_split = solution_str.split("Assistant:")
    response = solution_str_split[1]
    valid_template, reason = validate_template_format(response)
    if not valid_template:
        print("-"*100)
        print("格式错误 reward score: ", 0)
        print("-"*100)
        return 0, f'格式错误: {reason}'

    if response.endswith("<|endoftext|>"):
        response = response[:-len("<|endoftext|>")]
    else:
        print("-"*100)
        print("超出长度限制 reward score: ", 0)
        print("-"*100)
        return 0, f'超出长度限制'

    answer_part = extract_answer(response)
    if answer_part is not None:
        try:
            answer = remove_boxed(last_boxed_only_string(answer_part))
        except Exception as e:
            print("-"*100)
            print("提取boxed内容错误 reward score: ", 0)
            # print("predicted answer: ", response)
            print("-"*100)
            return 0, f'提取boxed内容错误: {e}'
    else:
        print("-"*100)
        print("无法提取答案 reward score: ", 0)
        # print("predicted answer: ", response)
        print("-"*100)
    if answer.lower() == ground_truth.lower():
        print("-"*100)
        print("reward score: ", 1)
        print("predicted answer: ", answer)
        print("ground truth: ", ground_truth)
        print("-"*100)
        return 1, f'答案正确: {answer}'
        
    # if ground_truth.lower() in answer.lower(): # dgt use cem
    #     return 1, f'CEM答案正确: {answer}'
    else:
        print("-"*100)
        print("答案错误但格式正确 reward score: ", 0.1)
        print("predicted answer: ", answer)
        # print("ground truth: ", ground_truth)
        print("-"*100)
        return 0.1, f'答案错误但格式正确: {answer}'


def extract_answer_dgt(solution_str) -> float:
    """
    计算答案的得分
    检查格式是否正确，答案是否与标准答案匹配
    返回: (得分, 原因说明)
    """
    answer_part = solution_str
    if "<answer>" in answer_part:
        answer_part = extract_answer(answer_part)
        if answer_part is not None:
            if "\\boxed{" in answer_part:
                answer = remove_boxed(last_boxed_only_string(answer_part))
            else:
                answer = answer_part
        else:
            answer = solution_str
            print("无法提取答案")

    elif "<answer>" not in answer_part:
        if "\\boxed{" in answer_part:
            answer = remove_boxed(last_boxed_only_string(answer_part))
        else:
            answer = answer_part

    else:
        answer = solution_str
        print("无法提取答案")
    return answer

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution

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

def extract_answer(str):
    if '<answer>' in str:
        start = str.index('<answer>')
        str = str[start + len('<answer>'):]
    if '</answer>' in str:
        end = str.index('</answer>')
        str = str[:end]
    if '\\boxed{' in str:
        start = str.index('\\boxed{')
        str = str[start + len('\\boxed{'):]
    if '}' in str:
        end = str.index('}')
        str = str[:end]
    return str.strip()