from tracemalloc import start
from unittest import result
from transformers import AutoTokenizer, AutoModel
from vllm import LLM
from collections import Counter
import re
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

def load_model(config):
    if 'e5' in config['model_path'].lower():
        model = AutoModel.from_pretrained(config['model_path'], trust_remote_code=True)
    else:
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

def cal_tool_use(input):
    count = 0
    
    python_start = 0
    while True:
        python_start = input.find("<python>", python_start)
        if python_start == -1:
            break
        python_end = input.find("</python>", python_start)
        if python_end == -1:
            break
        count += 1
        python_start = python_end + len('</python>')  
    
    search_start = 0
    while True:
        search_start = input.find("<search>", search_start)
        if search_start == -1:
            break
        search_end = input.find("</search>", search_start)
        if search_end == -1:
            break
        count += 1
        search_start = search_end + len('</search>')
    
    code_start = 0
    while True:
        code_start = input.find("```python", code_start)
        if code_start == -1:
            break
        code_end = input.find("```", code_start + len('```python'))
        if code_end == -1:
            break
        count += 1
        code_start = code_end + len('```') 
    
    return count

def cal_f1_score(string1, string2):
    if string1 is None or string2 is None:
        return 0.0
    string1 = string1.lower().split()
    string2 = string2.lower().split()
    counter1 = Counter(string1)
    counter2 = Counter(string2)
    common = counter1 & counter2
    if not common:
        return 0.0
    precision = sum(common.values()) / sum(counter1.values())
    recall = sum(common.values()) / sum(counter2.values())
    if precision + recall == 0:
        return 0.0
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def remove_tool_use(input):
    result = input
    start_tag = "<result>"
    end_tag = "</result>"
    while start_tag in result:
        # Find the index of the next <result> tag
        start_idx = result.find(start_tag)
        if start_idx == -1:
            break  # No more start tags found
        # Find the index of the corresponding </result> tag
        end_idx = result.find(end_tag, start_idx)
        if end_idx == -1:
            break  # No matching end tag, stop to avoid infinite loop
        # Include the length of </result> to remove the end tag as well
        end_idx += len(end_tag)
        # Remove the substring from start_idx to end_idx
        result = result[:start_idx] + result[end_idx:]
    return result

def remove_tool_use_torl(input):
    result = input
    start_tag = "```output"
    end_tag = "```"
    while start_tag in result:
        # Find the index of the next ```output tag
        start_idx = result.find(start_tag)
        if start_idx == -1:
            break  # No more start tags found
        # Find the index of the corresponding ``` tag
        end_idx = result.find(end_tag, start_idx)
        if end_idx == -1:
            break  # No matching end tag, stop to avoid infinite loop
        # Include the length of ``` to remove the end tag as well
        end_idx += len(end_tag)
        # Remove the substring from start_idx to end_idx
        result = result[:start_idx] + result[end_idx:]
    return result

def validate_format(text: str):
    # check if <think></think>, <answer></answer> is paired
    if text.count('<think>') != text.count('</think>'):
        return False
    
    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False
    
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False   
    
    # check the order of search/result and new logic for </think> before <search>
    current_pos = 0
    while True:
        search_pos = text.find('<search>', current_pos)
        if search_pos == -1:
            break
            
        result_pos = text.find('<result>', search_pos)
        search_end_pos = text.find('</search>', search_pos)
        result_end_pos = text.find('</result>', result_pos)
        
        if -1 in (result_pos, search_end_pos, result_end_pos):
            return False
            
        if not (search_pos < search_end_pos < result_pos < result_end_pos):
            return False
            
        # New logic: check if </think> is immediately before <search>
        text_before_search = text[:search_pos].rstrip()
        if not text_before_search.endswith('</think>'):
            return False
            
        # New logic: check if </result> is followed by <think> or <answer>
        text_after_result = text[result_end_pos + len('</result>'):].lstrip()
        if not (text_after_result.startswith('<think>') or text_after_result.startswith('<answer>')):
            return False
            
        current_pos = result_end_pos
    
    # check the order of python/result and new logic for </think> before <python>
    current_pos = 0
    while True:
        python_pos = text.find('<python>', current_pos)
        if python_pos == -1:
            break
            
        result_pos = text.find('<result>', python_pos)
        python_end_pos = text.find('</python>', python_pos)
        result_end_pos = text.find('</result>', result_pos)
        
        if -1 in (result_pos, python_end_pos, result_end_pos):
            return False
            
        if not (python_pos < python_end_pos < result_pos < result_end_pos):
            return False
            
        # New logic: check if </think> is immediately before <python>
        text_before_python = text[:python_pos].rstrip()
        if not text_before_python.endswith('</think>'):
            return False
            
        # New logic: check if </result> is followed by <think> or <answer>
        text_after_result = text[result_end_pos + len('</result>'):].lstrip()
        if not (text_after_result.startswith('<think>') or text_after_result.startswith('<answer>')):
            return False
            
        current_pos = result_end_pos
    
    # check if \boxed{} is in the answer
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False
    
    # check if any of the special tags appear inside the answer content
    special_tags = ['<think>', '</think>', '<python>', '</python>', '<search>', '</search>']
    for tag in special_tags:
        if tag in answer_content:
            return False
    
    return True

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

def contains_chinese(text):
    return bool(re.search('[\u4e00-\u9fff]', text))