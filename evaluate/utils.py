from ast import arg, parse
from dis import Instruction
import enum
import json 
from math import fabs
from operator import concat
import random
import math
import ujson

from tqdm import tqdm
import argparse
import re
import time
import datetime
import requests
from transformers import AutoTokenizer
from typing import List, Dict, Optional, final, Union
from vllm import LLM, SamplingParams

def search(query: str):
    if query == '':
        return 'invalid query'

    url = f'http://0.0.0.0:1243/search'
    data = {'query': query, 'top_n': 4}
    response = requests.post(url, json=data)
    retrieval_text = ''
    for line in response.json():
        retrieval_text += f"{line['contents']}\n\n"
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

def batch_search2(query: Union[str, List[str]], top_n=4) -> List[str]:
    if len(query) == 0:
        return 'invalid query'

    url = f'http://0.0.0.0:1243/batch_search'
    if isinstance(query, str):
        query = [query]
    data = {'query': query, 'top_n': top_n}
    response = requests.post(url, json=data)
    
    return response.json()

def extract_search_content(text: str) -> str:
    try:
        start_tag = '<search>'
        end_tag = '</search>'
        assert text.strip().endswith(end_tag)
        end_pos = text.rindex(end_tag)
        start_pos = text.rindex(start_tag, 0, end_pos)
        return text[start_pos + len(start_tag):end_pos].strip()
    except ValueError:
        return ""

def validate_template_format(text: str) -> tuple[bool, str]:
    # 检查 <think></think> 标签是否成对出现
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> 标签不成对"
    
    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "缺少 <think> 或 </think> 标签"
    
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> 或 </answer> 标签出现次数不为1"        
    
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
    
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> 必须在 </answer> 之前"
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "答案缺少 \\boxed{} 格式"
    
    return True, "格式正确"

def extract_answer(text: str):
    text = text.strip()

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    
    return match.group(1)

def remove_boxed(s):
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
        start_tag = '<python>'
        end_tag = '</python>'
        assert text.strip().endswith(end_tag)
        end_pos = text.rindex(end_tag)
        start_pos = text.rindex(start_tag, 0, end_pos)
        return text[start_pos + len(start_tag):end_pos].strip()
    except ValueError:
        return ""


def last_boxed_only_string(string):
    try:
        idx = string.rfind("\\boxed")
    except:
        import pdb; pdb.set_trace()
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