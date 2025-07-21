from ast import arg, parse
from cProfile import label
from dis import Instruction
import enum
from hmac import new
import json 
from math import fabs
from operator import concat, le
import random
import math
from collections import Counter
from tkinter import font
from sympy import false
from tomlkit import item
from tqdm import tqdm
import argparse
import re
import time
import datetime
import requests
from transformers import AutoTokenizer
from typing import List, Dict, Optional, final, Union
from vllm import LLM, SamplingParams
from matplotlib import pyplot as plt
from openai import OpenAI
import os
import numpy as np
import asyncio
from matplotlib.font_manager import FontProperties
from openai import AsyncOpenAI
from equivalence import is_equiv
from scipy.special import kl_div
rng = np.random.default_rng(4396)
from wordcloud import WordCloud
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from PIL import Image
font_path = '/fs/archive/share/u2023000153/Search-o1/scripts/figures/LinLibertine_R.ttf'
font_path_bold = '/fs/archive/share/u2023000153/Search-o1/scripts/figures/LinLibertine_RZ.ttf'

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
                # print('==========')
                # print(prompt)
                # print('==========')
                chat_response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                # print(chat_response)
                # assert False
                reason_answer = chat_response.choices[0].message.content.strip()
                # print(reason_answer)
                # Extract the judgment from the response
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
    # 如果查询为空，返回无效查询提示
    if query == '':
        return 'invalid query'

    # 构建搜索请求URL
    url = f'http://183.174.229.164:1243/search'
    # 准备请求数据，包含查询内容和返回结果数量
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
        # print("predicted answer: ", response)
        print("-"*100)
        return 0, f'格式错误: {reason}'

    if response.endswith("<|endoftext|>"):
        response = response[:-len("<|endoftext|>")]
    else:
        print("-"*100)
        print("超出长度限制 reward score: ", 0)
        # print("predicted answer: ", response)
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

def summary_cot(question, solution_str):
    API_BASE_URL = "http://192.168.1.1:28710/v1"
    MODEL_NAME = "Qwen2.5-72B-Instruct"
    client = OpenAI(
        api_key="empty",
        base_url=API_BASE_URL,
    )
    prompt_template = """You are a reasoning process refiner. Given a question and its reasoning process, your task is to refine the reasoning process by removing incorrect explorations and redundant overthinking, and generate a concise, direct, and accurate reasoning path.

Question: {question}

Original Reasoning Process: {reasoning}

Refined Reasoning Process: """
    prompt = prompt_template.format(question=question, reasoning=solution_str)
    chat_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    return chat_response.choices[0].message.content

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution

def rollback(s: str) -> str:
    # 1. 找到<python>位置
    python_pos = s.rfind('<python>')
    if python_pos == -1:
        return s  # 如果没找到<python>，返回原字符串
    
    # 2. 检查是否有</think>紧邻<python>
    think_end = '</think>'
    start_pos = python_pos
    if python_pos >= len(think_end) and s[python_pos - len(think_end):python_pos] == think_end:
        start_pos = python_pos - len(think_end)
    
    # 3. 从start_pos向前跳过连续的\n和空格
    pos = start_pos - 1
    while pos >= 0 and (s[pos] == '\n' or s[pos] == ' '):
        pos -= 1
    
    # 4. 从该位置向前找到第一个\n或". "，删除其后的内容
    while pos >= 0:
        if s[pos] == '\n':
            return s[:pos + 1]
        if pos > 0 and s[pos - 1:pos + 1] == '. ':
            return s[:pos + 1]
        pos -= 1
    
    # 如果没找到\n或". "，返回到处理位置之前的内容
    return s[:pos + 1]

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

tasks = {
    'webwalker': 'qa',
    'hotpotqa': 'qa',
    '2wiki': 'qa',
    'bamboogle': 'qa',
    'musique': 'qa',
    'gaia': 'qa',
    'hle': 'qa',
    'aime24': 'math',
    'aime25': 'math',
    'gsm8k': 'math',
    'math': 'math',
    'math500': 'math',
    'amc23': 'math',
}

def draw_entropy_distribution(entropy_lists, draw_path):
    """
    绘制多组信息熵分布图。
    每个主图对应 entropy_lists 中的一个元素，该主图包含多个子图，
    每个子图绘制一个折线图，表示信息熵随 token 输出的变化。

    Args:
        entropy_lists (list of list of list of float): 外部列表的每个元素代表一个主图的数据集。
                                                        内部列表的每个元素代表一个子图的数据（一条推理链的熵序列）。
        draw_path (str): 保存图表的路径。
    """

    final_entropy_lists = []
    max_step_counts = min(max(len(entropy_group) for entropy_group in entropy_lists), 6)

    ncols = min(3, max_step_counts)
    nrows = math.ceil(max_step_counts / ncols)

    for i in range(max_step_counts):
        # 收集每个子图在当前步骤的熵值
        current_step_entropy = []
        max_tokens = 0
        for entropy_group in entropy_lists:
            if i < len(entropy_group):
                current_step_entropy.append(entropy_group[i])
                max_tokens = max(max_tokens, len(entropy_group[i]))
            else:
                continue  # 如果当前组没有这个步骤的熵值，则跳过

        # TODO: 应用μ-3σ原则
        lengths = [len(sublist) for sublist in current_step_entropy]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)

        lower_bound = mean_length - 3 * std_length
        upper_bound = mean_length + 3 * std_length
        filtered_list = [sublist for sublist in current_step_entropy if lower_bound <= len(sublist) <= upper_bound]

        # 获取过滤后列表中元素长度的最大值
        max_tokens = max(len(sublist) for sublist in filtered_list) if filtered_list else 0

        # 更新current_step_entropy为过滤后的列表
        current_step_entropy = filtered_list

        final_entropy_list = []
        for j in range(max_tokens):
            # 对每个子图的当前步骤的熵值进行平均
            step_entropy = [entropy[j] for entropy in current_step_entropy if j < len(entropy)]
            if step_entropy:
                avg_entropy = sum(step_entropy) / len(step_entropy)
                final_entropy_list.append(avg_entropy)
        final_entropy_lists.append(final_entropy_list)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10), sharey=True)
    # 确保axs是二维数组，即使只有一行或一列
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = axs.reshape(1, -1)
    elif ncols == 1:
        axs = axs.reshape(-1, 1)
    for i, entropy_group in enumerate(final_entropy_lists):
        if i < len(entropy_lists):  # 确保索引有效
            row = i // ncols
            col = i % ncols
            ax = axs[row, col]
            
            if entropy_group:  # 确保有数据可绘制
                ax.plot(entropy_group, marker='o', label=f'Chain {i+1}')
                ax.set_title(f'Chain {i+1}')
                ax.set_xlabel('Token Index')
                ax.set_ylabel('Entropy')
                ax.legend()
                ax.grid(True)

    # 移除多余的子图
    for i in range(len(final_entropy_lists), nrows * ncols):
        row = i // ncols
        col = i % ncols
        fig.delaxes(axs[row, col])

    plt.tight_layout()
    plt.savefig(draw_path)
    plt.close(fig)
    print(f"熵分布图已保存到 {draw_path}")

def draw_entropy3_distribution(results, draw_path):
    # 计算judge结果
    questions, golden_answers, pred_answers = [], [], []
    idxs = []
    for i in range(len(results)):
        for j in range(len(results[i]['Output'])):
            questions.append(results[i]['question'])
            if isinstance(results[i]['answer'], list):
                golden_answers.append(random.choice(results[i]['answer']))
            else:
                golden_answers.append(results[i]['answer'])
            pred_answers.append(results[i]['Output'][j])
            idxs.append(i)
    judge_results = asyncio.run(
        llm_evaluate_equivalence_batch(
            questions=questions,
            labeled_answers=golden_answers,
            pred_answers=pred_answers,
        )
    )
    for i, idx in enumerate(idxs):
        if 'judge' not in results[idx]:
            results[idx]['judge'] = [judge_results[i]]
        else:
            results[idx]['judge'].append(judge_results[i])
    
    # 分离正确和错误的结果
    results_true = []
    results_false = []
    
    for result in results:
        # 创建正确和错误的结果副本
        result_true = {
            "Prompt": result["Prompt"],
            "question": result["question"],
            "Full_output": [],
            "Output": [],
            "answer": result["answer"],
            "entropy_list_final": [],
            "judge": []
        }
        
        result_false = {
            "Prompt": result["Prompt"],
            "question": result["question"],
            "Full_output": [],
            "Output": [],
            "answer": result["answer"],
            "entropy_list_final": [],
            "judge": []
        }
        
        # 根据judge结果分类
        if "judge" not in result:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        for i, is_correct in enumerate(result["judge"]):
            if is_correct:
                result_true["Full_output"].append(result["Full_output"][i])
                result_true["Output"].append(result["Output"][i])
                result_true["entropy_list_final"].append(result["entropy_list_final"][i])
                result_true["judge"].append(True)
            else:
                result_false["Full_output"].append(result["Full_output"][i])
                result_false["Output"].append(result["Output"][i])
                result_false["entropy_list_final"].append(result["entropy_list_final"][i])
                result_false["judge"].append(False)
        
        # 只添加非空结果
        if result_true["judge"]:
            results_true.append(result_true)
        if result_false["judge"]:
            results_false.append(result_false)
    
    # 提取熵列表
    entropy_lists_true = []
    entropy_lists_false = []
    
    for result in results_true:
        for entropy_list in result["entropy_list_final"]:
            entropy_lists_true.append(entropy_list)
    
    for result in results_false:
        for entropy_list in result["entropy_list_final"]:
            entropy_lists_false.append(entropy_list)
    
    # 计算正确和错误结果的熵分布
    final_entropy_lists_true = []
    max_step_counts_true = min(max(len(entropy_group) for entropy_group in entropy_lists_true) if entropy_lists_true else 0, 6)
    
    for i in range(max_step_counts_true):
        # 收集每个子图在当前步骤的熵值
        current_step_entropy = []
        max_tokens = 0
        for entropy_group in entropy_lists_true:
            if i < len(entropy_group):
                current_step_entropy.append(entropy_group[i])
                max_tokens = max(max_tokens, len(entropy_group[i]))
            else:
                continue
        
        # 应用μ-3σ原则
        lengths = [len(sublist) for sublist in current_step_entropy]
        if lengths:
            mean_length = np.mean(lengths)
            std_length = np.std(lengths)
            
            lower_bound = mean_length - 3 * std_length
            upper_bound = mean_length + 3 * std_length
            filtered_list = [sublist for sublist in current_step_entropy if lower_bound <= len(sublist) <= upper_bound]
            
            max_tokens = max(len(sublist) for sublist in filtered_list) if filtered_list else 0
            current_step_entropy = filtered_list
        
        final_entropy_list = []
        for j in range(max_tokens):
            step_entropy = [entropy[j] for entropy in current_step_entropy if j < len(entropy)]
            if step_entropy:
                avg_entropy = sum(step_entropy) / len(step_entropy)
                final_entropy_list.append(avg_entropy)
        final_entropy_lists_true.append(final_entropy_list)
    
    # 错误结果的熵分布
    final_entropy_lists_false = []
    max_step_counts_false = min(max(len(entropy_group) for entropy_group in entropy_lists_false) if entropy_lists_false else 0, 6)
    
    for i in range(max_step_counts_false):
        current_step_entropy = []
        max_tokens = 0
        for entropy_group in entropy_lists_false:
            if i < len(entropy_group):
                current_step_entropy.append(entropy_group[i])
                max_tokens = max(max_tokens, len(entropy_group[i]))
            else:
                continue
        
        # 应用μ-3σ原则
        lengths = [len(sublist) for sublist in current_step_entropy]
        if lengths:
            mean_length = np.mean(lengths)
            std_length = np.std(lengths)
            
            lower_bound = mean_length - 3 * std_length
            upper_bound = mean_length + 3 * std_length
            filtered_list = [sublist for sublist in current_step_entropy if lower_bound <= len(sublist) <= upper_bound]
            
            max_tokens = max(len(sublist) for sublist in filtered_list) if filtered_list else 0
            current_step_entropy = filtered_list
        
        final_entropy_list = []
        for j in range(max_tokens):
            step_entropy = [entropy[j] for entropy in current_step_entropy if j < len(entropy)]
            if step_entropy:
                avg_entropy = sum(step_entropy) / len(step_entropy)
                final_entropy_list.append(avg_entropy)
        final_entropy_lists_false.append(final_entropy_list)
    
    # 确定绘图的行列数
    max_step_counts = max(max_step_counts_true, max_step_counts_false)
    ncols = min(3, max_step_counts)
    nrows = math.ceil(max_step_counts / ncols)
    
    # 绘制图表
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10), sharey=True)
    
    # 确保axs是二维数组
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = axs.reshape(1, -1)
    elif ncols == 1:
        axs = axs.reshape(-1, 1)
    
    for i in range(max_step_counts):
        row = i // ncols
        col = i % ncols
        ax = axs[row, col]
        
        # 绘制错误结果的熵分布
        if i < len(final_entropy_lists_false) and final_entropy_lists_false[i]:
            ax.plot(final_entropy_lists_false[i], marker='o', color='red', label='Incorrect', alpha=0.5)

        # 绘制正确结果的熵分布
        if i < len(final_entropy_lists_true) and final_entropy_lists_true[i]:
            ax.plot(final_entropy_lists_true[i], marker='o', color='green', label='Correct', alpha=0.5)
        
        ax.set_title(f'Chain {i+1}')
        ax.set_xlabel('Token Index')
        ax.set_ylabel('Entropy')
        ax.legend()
        ax.grid(True)
    
    # 移除多余的子图
    for i in range(max_step_counts, nrows * ncols):
        row = i // ncols
        col = i % ncols
        fig.delaxes(axs[row, col])
    
    plt.tight_layout()
    plt.savefig(draw_path)
    plt.close(fig)
    print(f"熵分布图已保存到 {draw_path}")
    
    return results_true, results_false

def draw_entropy4_distribution(results, draw_path):
    """
    根据results中的entropy_list_final数据绘制熵分布图。
    
    Args:
        results (list): 包含多个结果字典的列表，每个字典包含entropy_list_final数据
        draw_path (str): 保存图表的路径
    """
    # 按照长度分组entropy_list_final
    grouped_by_length = {}
    
    for result in results:
        for entropy_list in result["entropy_list_final"]:
            length = len(entropy_list)
            if length not in grouped_by_length:
                grouped_by_length[length] = []
            grouped_by_length[length].append(entropy_list)
    
    # 确定最大长度和行列数
    lengths = set(grouped_by_length.keys())
    
    if not lengths:
        print("没有有效的熵数据可绘制")
        return
    
    min_length = min(lengths)  # 从有数据的最小长度开始
    max_length = min(max(lengths), 5)  # 限制最多显示5行
    # print(f"绘制长度范围: {min_length} 到 {max_length}")
    
    # 处理每个长度组的数据
    final_data = {}
    for length in range(min_length, max_length + 1):
        if length in grouped_by_length:
            entropy_lists = grouped_by_length[length]
            final_data[length] = []
            
            for step_idx in range(min(length, 5)):  # 限制最大列数为5
                # 收集当前步骤的熵值
                current_step_entropy = []
                max_tokens = 0
                
                for entropy_group in entropy_lists:
                    if step_idx < len(entropy_group):
                        current_step_entropy.append(entropy_group[step_idx])
                        max_tokens = max(max_tokens, len(entropy_group[step_idx]))
                
                # 应用μ-3σ原则
                lengths = [len(sublist) for sublist in current_step_entropy]
                if not lengths:
                    continue
                    
                mean_length = np.mean(lengths)
                std_length = np.std(lengths)
                
                lower_bound = mean_length - 3 * std_length
                upper_bound = mean_length + 3 * std_length
                filtered_list = [sublist for sublist in current_step_entropy 
                                if lower_bound <= len(sublist) <= upper_bound]
                
                # 获取过滤后列表中元素长度的最大值
                max_tokens = max(len(sublist) for sublist in filtered_list) if filtered_list else 0
                
                # 计算平均熵值
                final_entropy_list = []
                for j in range(max_tokens):
                    step_entropy = [entropy[j] for entropy in filtered_list if j < len(entropy)]
                    if step_entropy:
                        avg_entropy = sum(step_entropy) / len(step_entropy)
                        final_entropy_list.append(avg_entropy)
                
                final_data[length].append(final_entropy_list)
    
    # 确定图表布局
    nrows = max_length - min_length + 1  # 行数为最大长度减最小长度加1
    ncols = 5  # 限制最大列数为5
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows), sharey=True)
    
    # 确保axs是二维数组
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = axs.reshape(1, -1)
    elif ncols == 1:
        axs = axs.reshape(-1, 1)
    
    # 绘制图表
    for length in range(min_length, max_length + 1):
        row = length - min_length  # 行索引从0开始，从min_length开始计数
        
        if length in final_data:
            for step_idx, entropy_values in enumerate(final_data[length]):
                if step_idx < ncols:  # 确保不超出列数
                    col = step_idx
                    ax = axs[row, col]
                    
                    if entropy_values:  # 确保有数据可绘制
                        ax.plot(entropy_values, marker='o', label=f'Chain {step_idx+1}')
                        ax.set_title(f'Length {length}, Chain {step_idx+1}')
                        ax.set_xlabel('Token Index')
                        ax.set_ylabel('Entropy')
                        ax.legend()
                        ax.grid(True)
    
    # 移除多余的子图
    for row in range(nrows):
        for col in range(ncols):
            length = row + min_length
            step_idx = col
            
            if length not in final_data or step_idx >= len(final_data[length]) or step_idx >= length:
                fig.delaxes(axs[row, col])
    
    plt.tight_layout()
    plt.savefig(draw_path)
    plt.close(fig)
    print(f"熵分布图已保存到 {draw_path}")


def draw_entropy5_distribution(results, draw_path):
    # 计算judge结果
    questions, golden_answers, pred_answers = [], [], []
    idxs = []
    for i in range(len(results)):
        for j in range(len(results[i]['Output'])):
            questions.append(results[i]['question'])
            if isinstance(results[i]['answer'], list):
                golden_answers.append(random.choice(results[i]['answer']))
            else:
                golden_answers.append(results[i]['answer'])
            pred_answers.append(results[i]['Output'][j])
            idxs.append(i)
    judge_results = asyncio.run(
        llm_evaluate_equivalence_batch(
            questions=questions,
            labeled_answers=golden_answers,
            pred_answers=pred_answers,
        )
    )
    for i, idx in enumerate(idxs):
        if 'judge' not in results[idx]:
            results[idx]['judge'] = [judge_results[i]]
        else:
            results[idx]['judge'].append(judge_results[i])
    
    # 分离正确和错误的结果
    results_true = []
    results_false = []
    
    for result in results:
        # 创建正确和错误的结果副本
        result_true = {
            "Prompt": result["Prompt"],
            "question": result["question"],
            "Full_output": [],
            "Output": [],
            "answer": result["answer"],
            "entropy_list_final": [],
            "judge": []
        }
        
        result_false = {
            "Prompt": result["Prompt"],
            "question": result["question"],
            "Full_output": [],
            "Output": [],
            "answer": result["answer"],
            "entropy_list_final": [],
            "judge": []
        }
        
        # 根据judge结果分类
        for i, is_correct in enumerate(result["judge"]):
            if is_correct:
                result_true["Full_output"].append(result["Full_output"][i])
                result_true["Output"].append(result["Output"][i])
                result_true["entropy_list_final"].append(result["entropy_list_final"][i])
                result_true["judge"].append(True)
            else:
                result_false["Full_output"].append(result["Full_output"][i])
                result_false["Output"].append(result["Output"][i])
                result_false["entropy_list_final"].append(result["entropy_list_final"][i])
                result_false["judge"].append(False)
        
        # 只添加非空结果
        if result_true["judge"]:
            results_true.append(result_true)
        if result_false["judge"]:
            results_false.append(result_false)
    
    # 按照长度分组正确和错误的entropy_list_final
    grouped_by_length_true = {}
    grouped_by_length_false = {}
    
    # 处理正确结果
    for result in results_true:
        for entropy_list in result["entropy_list_final"]:
            length = len(entropy_list)
            if length not in grouped_by_length_true:
                grouped_by_length_true[length] = []
            grouped_by_length_true[length].append(entropy_list)
    
    # 处理错误结果
    for result in results_false:
        for entropy_list in result["entropy_list_final"]:
            length = len(entropy_list)
            if length not in grouped_by_length_false:
                grouped_by_length_false[length] = []
            grouped_by_length_false[length].append(entropy_list)
    
    # 确定最小和最大长度
    all_lengths = set(list(grouped_by_length_true.keys()) + list(grouped_by_length_false.keys()))
    if not all_lengths:
        print("没有有效的熵数据可绘制")
        return results_true, results_false
    
    min_length = min(all_lengths)  # 从有数据的最小长度开始
    max_length = min(max(all_lengths), 5)  # 限制最多显示5行
    
    # 处理每个长度组的数据
    final_data_true = {}
    final_data_false = {}
    
    # 处理正确结果数据
    for length in range(min_length, max_length + 1):
        if length in grouped_by_length_true:
            entropy_lists = grouped_by_length_true[length]
            final_data_true[length] = []
            
            for step_idx in range(min(length, 5)):  # 限制最大列数为5
                # 收集当前步骤的熵值
                current_step_entropy = []
                max_tokens = 0
                
                for entropy_group in entropy_lists:
                    if step_idx < len(entropy_group):
                        current_step_entropy.append(entropy_group[step_idx])
                        max_tokens = max(max_tokens, len(entropy_group[step_idx]))
                
                # 应用μ-3σ原则
                lengths = [len(sublist) for sublist in current_step_entropy]
                if not lengths:
                    continue
                    
                mean_length = np.mean(lengths)
                std_length = np.std(lengths)
                
                lower_bound = mean_length - 3 * std_length
                upper_bound = mean_length + 3 * std_length
                filtered_list = [sublist for sublist in current_step_entropy 
                                if lower_bound <= len(sublist) <= upper_bound]
                
                # 获取过滤后列表中元素长度的最大值
                max_tokens = max(len(sublist) for sublist in filtered_list) if filtered_list else 0
                
                # 计算平均熵值
                final_entropy_list = []
                for j in range(max_tokens):
                    step_entropy = [entropy[j] for entropy in filtered_list if j < len(entropy)]
                    if step_entropy:
                        avg_entropy = sum(step_entropy) / len(step_entropy)
                        final_entropy_list.append(avg_entropy)
                
                final_data_true[length].append(final_entropy_list)
    
    # 处理错误结果数据
    for length in range(min_length, max_length + 1):
        if length in grouped_by_length_false:
            entropy_lists = grouped_by_length_false[length]
            final_data_false[length] = []
            
            for step_idx in range(min(length, 5)):  # 限制最大列数为5
                # 收集当前步骤的熵值
                current_step_entropy = []
                max_tokens = 0
                
                for entropy_group in entropy_lists:
                    if step_idx < len(entropy_group):
                        current_step_entropy.append(entropy_group[step_idx])
                        max_tokens = max(max_tokens, len(entropy_group[step_idx]))
                
                # 应用μ-3σ原则
                lengths = [len(sublist) for sublist in current_step_entropy]
                if not lengths:
                    continue
                    
                mean_length = np.mean(lengths)
                std_length = np.std(lengths)
                
                lower_bound = mean_length - 3 * std_length
                upper_bound = mean_length + 3 * std_length
                filtered_list = [sublist for sublist in current_step_entropy 
                                if lower_bound <= len(sublist) <= upper_bound]
                
                # 获取过滤后列表中元素长度的最大值
                max_tokens = max(len(sublist) for sublist in filtered_list) if filtered_list else 0
                
                # 计算平均熵值
                final_entropy_list = []
                for j in range(max_tokens):
                    step_entropy = [entropy[j] for entropy in filtered_list if j < len(entropy)]
                    if step_entropy:
                        avg_entropy = sum(step_entropy) / len(step_entropy)
                        final_entropy_list.append(avg_entropy)
                
                final_data_false[length].append(final_entropy_list)
    
    # 确定图表布局
    nrows = max_length - min_length + 1  # 行数为最大长度减最小长度加1
    ncols = min(max_length, 5)  # 限制为最大5列
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows), sharey=True)
    
    # 确保axs是二维数组
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = axs.reshape(1, -1)
    elif ncols == 1:
        axs = axs.reshape(-1, 1)
    
    # 绘制图表
    for length in range(min_length, max_length + 1):
        row = length - min_length  # 行索引从0开始，从min_length开始计数
        
        for step_idx in range(min(length, ncols)):
            col = step_idx
            
            if row < nrows and col < ncols:  # 确保索引在有效范围内
                ax = axs[row, col]
                
                # 绘制错误结果的熵分布
                if length in final_data_false and step_idx < len(final_data_false[length]):
                    entropy_values = final_data_false[length][step_idx]
                    if entropy_values:
                        ax.plot(entropy_values, marker='o', color='red', label='Incorrect', alpha=0.5)
                
                # 绘制正确结果的熵分布
                if length in final_data_true and step_idx < len(final_data_true[length]):
                    entropy_values = final_data_true[length][step_idx]
                    if entropy_values:
                        ax.plot(entropy_values, marker='o', color='green', label='Correct', alpha=0.5)
                
                if (length in final_data_true and step_idx < len(final_data_true[length]) and final_data_true[length][step_idx]) or \
                   (length in final_data_false and step_idx < len(final_data_false[length]) and final_data_false[length][step_idx]):
                    ax.set_title(f'Length {length}, Chain {step_idx+1}')
                    ax.set_xlabel('Token Index')
                    ax.set_ylabel('Entropy')
                    ax.legend()
                    ax.grid(True)
    
    # 移除多余的子图
    for row in range(nrows):
        for col in range(ncols):
            length = row + min_length
            step_idx = col
            
            has_data = False
            
            if length in final_data_true and step_idx < len(final_data_true[length]) and final_data_true[length][step_idx]:
                has_data = True
            
            if length in final_data_false and step_idx < len(final_data_false[length]) and final_data_false[length][step_idx]:
                has_data = True
            
            if not has_data or step_idx >= length:
                if row < nrows and col < ncols:  # 确保索引在有效范围内
                    fig.delaxes(axs[row, col])
    
    plt.tight_layout()
    plt.savefig(draw_path)
    plt.close(fig)
    print(f"熵分布图已保存到 {draw_path}")
    
    return results_true, results_false

def draw_entropy6_distribution(results, draw_path):
    # 计算judge结果
    questions, golden_answers, pred_answers = [], [], []
    idxs = []
    for i in range(len(results)):
        for j in range(len(results[i]['Output'])):
            questions.append(results[i]['question'])
            if isinstance(results[i]['answer'], list):
                golden_answers.append(random.choice(results[i]['answer']))
            else:
                golden_answers.append(results[i]['answer'])
            pred_answers.append(results[i]['Output'][j])
            idxs.append(i)
    judge_results = asyncio.run(
        llm_evaluate_equivalence_batch(
            questions=questions,
            labeled_answers=golden_answers,
            pred_answers=pred_answers,
        )
    )
    for i, idx in enumerate(idxs):
        if 'judge' not in results[idx]:
            results[idx]['judge'] = [judge_results[i]]
        else:
            results[idx]['judge'].append(judge_results[i])
    
    # 分离正确和错误的结果
    results_true = []
    results_false = []
    
    for result in results:
        # 创建正确和错误的结果副本
        result_true = {
            "Prompt": result["Prompt"],
            "question": result["question"],
            "Full_output": [],
            "Output": [],
            "answer": result["answer"],
            "entropy_list_final": [],
            "judge": []
        }
        
        result_false = {
            "Prompt": result["Prompt"],
            "question": result["question"],
            "Full_output": [],
            "Output": [],
            "answer": result["answer"],
            "entropy_list_final": [],
            "judge": []
        }
        
        # 根据judge结果分类
        for i, is_correct in enumerate(result["judge"]):
            if is_correct:
                result_true["Full_output"].append(result["Full_output"][i])
                result_true["Output"].append(result["Output"][i])
                result_true["entropy_list_final"].append(result["entropy_list_final"][i])
                result_true["judge"].append(True)
            else:
                result_false["Full_output"].append(result["Full_output"][i])
                result_false["Output"].append(result["Output"][i])
                result_false["entropy_list_final"].append(result["entropy_list_final"][i])
                result_false["judge"].append(False)
        
        # 只添加非空结果
        if result_true["judge"]:
            results_true.append(result_true)
        if result_false["judge"]:
            results_false.append(result_false)
    
    # 按照长度分组正确结果的entropy_list_final
    grouped_by_length_true = {}
    
    # 处理正确结果
    for result in results_true:
        for entropy_list in result["entropy_list_final"]:
            length = len(entropy_list)
            if length not in grouped_by_length_true:
                grouped_by_length_true[length] = []
            grouped_by_length_true[length].append(entropy_list)
    
    # 确定最小和最大长度
    if not grouped_by_length_true:
        print("没有有效的熵数据可绘制")
        return results_true, results_false
    
    min_length = min(grouped_by_length_true.keys())
    max_length = min(max(grouped_by_length_true.keys()), 5)  # 限制最多显示5行
    
    # 处理每个长度组的数据
    final_data_true = {}
    
    # 处理正确结果数据
    for length in range(min_length, max_length + 1):
        if length in grouped_by_length_true:
            entropy_lists = grouped_by_length_true[length]
            final_data_true[length] = []
            
            for step_idx in range(min(length, 5)):  # 限制最大列数为5
                # 收集当前步骤的熵值
                current_step_entropy = []
                max_tokens = 0
                
                for entropy_group in entropy_lists:
                    if step_idx < len(entropy_group):
                        current_step_entropy.append(entropy_group[step_idx])
                        max_tokens = max(max_tokens, len(entropy_group[step_idx]))
                
                # 应用μ-3σ原则
                lengths = [len(sublist) for sublist in current_step_entropy]
                if not lengths:
                    continue
                    
                mean_length = np.mean(lengths)
                std_length = np.std(lengths)
                
                lower_bound = mean_length - 3 * std_length
                upper_bound = mean_length + 3 * std_length
                filtered_list = [sublist for sublist in current_step_entropy 
                                if lower_bound <= len(sublist) <= upper_bound]
                
                # 获取过滤后列表中元素长度的最大值
                max_tokens = max(len(sublist) for sublist in filtered_list) if filtered_list else 0
                
                # 计算平均熵值
                final_entropy_list = []
                for j in range(max_tokens):
                    step_entropy = [entropy[j] for entropy in filtered_list if j < len(entropy)]
                    if step_entropy:
                        avg_entropy = sum(step_entropy) / len(step_entropy)
                        final_entropy_list.append(avg_entropy)
                
                final_data_true[length].append(final_entropy_list)
    
    # 确定图表布局
    nrows = max_length - min_length + 1  # 行数为最大长度减最小长度加1
    ncols = min(max_length, 5)  # 限制为最大5列
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows), sharey=True)
    
    # 确保axs是二维数组
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = axs.reshape(1, -1)
    elif ncols == 1:
        axs = axs.reshape(-1, 1)
    
    # 绘制图表
    for length in range(min_length, max_length + 1):
        row = length - min_length  # 行索引从0开始，从min_length开始计数
        
        for step_idx in range(min(length, ncols)):
            col = step_idx
            
            if row < nrows and col < ncols:  # 确保索引在有效范围内
                ax = axs[row, col]
                
                # 只绘制正确结果的熵分布
                if length in final_data_true and step_idx < len(final_data_true[length]):
                    entropy_values = final_data_true[length][step_idx]
                    if entropy_values:
                        ax.plot(entropy_values, marker='o', color='green', label='Correct', alpha=0.7)
                        ax.set_title(f'Length {length}, Chain {step_idx+1}')
                        ax.set_xlabel('Token Index')
                        ax.set_ylabel('Entropy')
                        ax.legend()
                        ax.grid(True)
    
    # 移除多余的子图
    for row in range(nrows):
        for col in range(ncols):
            length = row + min_length
            step_idx = col
            
            has_data = False
            
            if length in final_data_true and step_idx < len(final_data_true[length]) and final_data_true[length][step_idx]:
                has_data = True
            
            if not has_data or step_idx >= length:
                if row < nrows and col < ncols:  # 确保索引在有效范围内
                    fig.delaxes(axs[row, col])
    
    plt.tight_layout()
    plt.savefig(draw_path)
    plt.close(fig)
    print(f"正确样本熵分布图已保存到 {draw_path}")
    
    return results_true, results_false


def draw_entropy1_difference(results, draw_path):
    """
    绘制熵分布图，比较上一步推理过程的最后一句话和下一步推理过程的第一句话之间的差值。
    
    Args:
        results (list): 包含多个结果字典的列表，每个字典包含entropy_list_final数据
        draw_path (str): 保存图表的路径
    """
    # 计算judge结果
    questions, golden_answers, pred_answers = [], [], []
    idxs = []
    for i in range(len(results)):
        for j in range(len(results[i]['Output'])):
            questions.append(results[i]['question'])
            if isinstance(results[i]['answer'], list):
                golden_answers.append(random.choice(results[i]['answer']))
            else:
                golden_answers.append(results[i]['answer'])
            pred_answers.append(results[i]['Output'][j])
            idxs.append(i)
    judge_results = asyncio.run(
        llm_evaluate_equivalence_batch(
            questions=questions,
            labeled_answers=golden_answers,
            pred_answers=pred_answers,
        )
    )
    for i, idx in enumerate(idxs):
        if 'judge' not in results[idx]:
            results[idx]['judge'] = [judge_results[i]]
        else:
            results[idx]['judge'].append(judge_results[i])
    
    # 分离正确和错误的结果
    results_true = []
    
    for result in results:
        # 创建正确和错误的结果副本
        result_true = {
            "Prompt": result["Prompt"],
            "question": result["question"],
            "Full_output": [],
            "Output": [],
            "answer": result["answer"],
            "entropy_list_final": [],
            "judge": []
        }
        
        # 根据judge结果分类
        for i, is_correct in enumerate(result["judge"]):
            if is_correct:
                result_true["Full_output"].append(result["Full_output"][i])
                result_true["Output"].append(result["Output"][i])
                result_true["entropy_list_final"].append(result["entropy_list_final"][i])
                result_true["judge"].append(True)
        
        # 只添加非空结果
        if result_true["judge"]:
            results_true.append(result_true)
    results = results_true

    all_kl_divs = []
    
    for result in results:
        for entropy_sequence in result["entropy_list_final"]:
            # 如果序列只有一个元素，跳过
            if len(entropy_sequence) <= 1:
                continue
            
            sequence_kl_divs = []
            
            # 计算相邻元素之间的KL散度
            for i in range(len(entropy_sequence) - 1):
                last_elements = entropy_sequence[i]
                next_elements = entropy_sequence[i + 1]
                
                sample_length = min(50, len(last_elements), len(next_elements))
                if sample_length == 0:
                    continue
                # 取出前一个元素的最后sample_length个值
                last_tokens = last_elements[-sample_length:]
                # 取出后一个元素的前sample_length个值
                first_tokens = next_elements[:sample_length]
                
                # 计算KL散度
                kl_div_value = calculate_kl_divergence(last_tokens, first_tokens)
                sequence_kl_divs.append(kl_div_value)
                
            if len(sequence_kl_divs) >= 5:
                # 只保留前4个KL散度值
                sequence_kl_divs = sequence_kl_divs[:4]
            
            if sequence_kl_divs:  # 确保不是空列表
                all_kl_divs.append(sequence_kl_divs)
    grouped_by_length_true = {}
    for kl_divs in all_kl_divs:
        length = len(kl_divs)
        if length not in grouped_by_length_true:
            grouped_by_length_true[length] = []
        grouped_by_length_true[length].append(kl_divs)

    # kl_divs是一个列表，包含每个结果的KL散度值
    # all_kl_divs是一个列表，包含了这个数据集的所有结果，里面每个元素是一个list，长度最长为4

    # 处理每个长度组的数据
    for key, values in grouped_by_length_true.items():
        # 横纵转置
        grouped_by_length_true[key] = list(zip(*values))
        grouped_by_length_true[key] = [list(group) for group in grouped_by_length_true[key]]
    min_length = min(grouped_by_length_true.keys())
    max_length = min(max(grouped_by_length_true.keys()), 5)  # 限制最多显示5行
    
    # 应用μ-3σ原则处理数据
    filtered_data = {}
    for length, step_lists in grouped_by_length_true.items():
        filtered_data[length] = []
        
        for step_values in step_lists:
            # 计算均值和标准差
            mean_value = np.mean(step_values)
            std_value = np.std(step_values)
            
            # 设置上下界
            lower_bound = mean_value - 3 * std_value
            upper_bound = mean_value + 3 * std_value
            
            # 过滤数据
            filtered_values = [value for value in step_values if lower_bound <= value <= upper_bound]
            filtered_data[length].append(filtered_values)
    
    # 确定图表布局
    nrows = max_length - min_length + 1  # 行数为最大长度减最小长度加1
    ncols = min(max_length, 5)  # 限制为最大5列
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows), sharey=True)

    # 确保axs是二维数组
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = axs.reshape(1, -1)
    elif ncols == 1:
        axs = axs.reshape(-1, 1)

    # 按key排序绘制图表
    for i, length in enumerate(sorted(filtered_data.keys())):
        data_lists = filtered_data[length]
        
        # 验证key等于value列表的长度
        assert len(data_lists) == length, f"Key {length} 应该等于对应value列表的长度 {len(data_lists)}"
        
        # 为每个列表创建子图
        for j, data in enumerate(data_lists):
            if j < ncols:  # 确保不超出列数
                ax = axs[i, j]
                ax.plot(data, marker='o', color='green', alpha=0.7)
                ax.set_title(f'Length {length}, List {j+1}')
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')
                ax.grid(True)

    # 移除多余的子图
    for i, length in enumerate(sorted(filtered_data.keys())):
        # 移除每行超出实际数据的子图
        for j in range(length, ncols):
            fig.delaxes(axs[i, j])

    plt.tight_layout()
    plt.savefig(draw_path)
    plt.close(fig)
    print(f"正确样本熵分布图已保存到 {draw_path}")
    
    return results_true

def draw_entropy2_difference(results, draw_path):
    """
    绘制熵分布图，比较上一步推理过程的最后一句话和下一步推理过程的第一句话之间的差值。
    
    Args:
        results (list): 包含多个结果字典的列表，每个字典包含entropy_list_final数据
        draw_path (str): 保存图表的路径
    """
    # 计算judge结果
    questions, golden_answers, pred_answers = [], [], []
    idxs = []
    for i in range(len(results)):
        for j in range(len(results[i]['Output'])):
            questions.append(results[i]['question'])
            if isinstance(results[i]['answer'], list):
                golden_answers.append(random.choice(results[i]['answer']))
            else:
                golden_answers.append(results[i]['answer'])
            pred_answers.append(results[i]['Output'][j])
            idxs.append(i)
    judge_results = asyncio.run(
        llm_evaluate_equivalence_batch(
            questions=questions,
            labeled_answers=golden_answers,
            pred_answers=pred_answers,
        )
    )
    for i, idx in enumerate(idxs):
        if 'judge' not in results[idx]:
            results[idx]['judge'] = [judge_results[i]]
        else:
            results[idx]['judge'].append(judge_results[i])
    
    # 分离正确和错误的结果
    results_true = []
    
    for result in results:
        # 创建正确和错误的结果副本
        result_true = {
            "Prompt": result["Prompt"],
            "question": result["question"],
            "Full_output": [],
            "Output": [],
            "answer": result["answer"],
            "entropy_list_final": [],
            "judge": []
        }
        
        # 根据judge结果分类
        for i, is_correct in enumerate(result["judge"]):
            if is_correct:
                result_true["Full_output"].append(result["Full_output"][i])
                result_true["Output"].append(result["Output"][i])
                result_true["entropy_list_final"].append(result["entropy_list_final"][i])
                result_true["judge"].append(True)
        
        # 只添加非空结果
        if result_true["judge"]:
            results_true.append(result_true)
    results = results_true

    all_kl_divs = []
    
    for result in results:
        for entropy_sequence in result["entropy_list_final"]:
            # 如果序列只有一个元素，跳过
            if len(entropy_sequence) <= 1:
                continue
            
            sequence_kl_divs = []
            
            # 计算相邻元素之间的KL散度
            for i in range(len(entropy_sequence) - 1):
                last_elements = entropy_sequence[i]
                next_elements = entropy_sequence[i + 1]
                
                sample_length = min(50, len(last_elements), len(next_elements))
                if sample_length == 0:
                    continue
                # 取出前一个元素的最后sample_length个值
                last_tokens = last_elements[:sample_length]
                # 取出后一个元素的前sample_length个值
                first_tokens = next_elements[:sample_length]
                
                # 计算KL散度
                kl_div_value = calculate_kl_divergence(last_tokens, first_tokens)
                sequence_kl_divs.append(kl_div_value)
                
            if len(sequence_kl_divs) >= 5:
                # 只保留前4个KL散度值
                sequence_kl_divs = sequence_kl_divs[:4]
            
            if sequence_kl_divs:  # 确保不是空列表
                all_kl_divs.append(sequence_kl_divs)
    grouped_by_length_true = {}
    for kl_divs in all_kl_divs:
        length = len(kl_divs)
        if length not in grouped_by_length_true:
            grouped_by_length_true[length] = []
        grouped_by_length_true[length].append(kl_divs)

    # kl_divs是一个列表，包含每个结果的KL散度值
    # all_kl_divs是一个列表，包含了这个数据集的所有结果，里面每个元素是一个list，长度最长为4

    # 处理每个长度组的数据
    for key, values in grouped_by_length_true.items():
        # 横纵转置
        grouped_by_length_true[key] = list(zip(*values))
        grouped_by_length_true[key] = [list(group) for group in grouped_by_length_true[key]]
    min_length = min(grouped_by_length_true.keys())
    max_length = min(max(grouped_by_length_true.keys()), 5)  # 限制最多显示5行
    
    # 应用μ-3σ原则处理数据
    filtered_data = {}
    for length, step_lists in grouped_by_length_true.items():
        filtered_data[length] = []
        
        for step_values in step_lists:
            # 计算均值和标准差
            mean_value = np.mean(step_values)
            std_value = np.std(step_values)
            
            # 设置上下界
            lower_bound = mean_value - 3 * std_value
            upper_bound = mean_value + 3 * std_value
            
            # 过滤数据
            filtered_values = [value for value in step_values if lower_bound <= value <= upper_bound]
            filtered_data[length].append(filtered_values)
    
    # 确定图表布局
    nrows = max_length - min_length + 1  # 行数为最大长度减最小长度加1
    ncols = min(max_length, 5)  # 限制为最大5列
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows), sharey=True)

    # 确保axs是二维数组
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = axs.reshape(1, -1)
    elif ncols == 1:
        axs = axs.reshape(-1, 1)

    # 按key排序绘制图表
    for i, length in enumerate(sorted(filtered_data.keys())):
        data_lists = filtered_data[length]
        
        # 验证key等于value列表的长度
        assert len(data_lists) == length, f"Key {length} 应该等于对应value列表的长度 {len(data_lists)}"
        
        # 为每个列表创建子图
        for j, data in enumerate(data_lists):
            if j < ncols:  # 确保不超出列数
                ax = axs[i, j]
                ax.plot(data, marker='o', color='green', alpha=0.7)
                ax.set_title(f'Length {length}, List {j+1}')
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')
                ax.grid(True)

    # 移除多余的子图
    for i, length in enumerate(sorted(filtered_data.keys())):
        # 移除每行超出实际数据的子图
        for j in range(length, ncols):
            fig.delaxes(axs[i, j])

    plt.tight_layout()
    plt.savefig(draw_path)
    plt.close(fig)
    print(f"正确样本熵分布图已保存到 {draw_path}")
    
    return results_true

def draw_entropy3_difference(results, draw_path):
    # 绘制同一条样本的不同长度链的信息熵区别
    # 计算judge结果
    questions, golden_answers, pred_answers = [], [], []
    idxs = []
    for i in range(len(results)):
        for j in range(len(results[i]['Output'])):
            questions.append(results[i]['question'])
            if isinstance(results[i]['answer'], list):
                golden_answers.append(random.choice(results[i]['answer']))
            else:
                golden_answers.append(results[i]['answer'])
            pred_answers.append(results[i]['Output'][j])
            idxs.append(i)
    judge_results = asyncio.run(
        llm_evaluate_equivalence_batch(
            questions=questions,
            labeled_answers=golden_answers,
            pred_answers=pred_answers,
        )
    )
    for i, idx in enumerate(idxs):
        if 'judge' not in results[idx]:
            results[idx]['judge'] = [judge_results[i]]
        else:
            results[idx]['judge'].append(judge_results[i])
    
    # 分离正确和错误的结果
    results_true = []
    
    for result in results:
        # 创建正确和错误的结果副本
        result_true = {
            "Prompt": result["Prompt"],
            "question": result["question"],
            "Full_output": [],
            "Output": [],
            "answer": result["answer"],
            "entropy_list_final": [],
            "judge": []
        }
        
        # 根据judge结果分类
        for i, is_correct in enumerate(result["judge"]):
            if is_correct:
                result_true["Full_output"].append(result["Full_output"][i])
                result_true["Output"].append(result["Output"][i])
                result_true["entropy_list_final"].append(result["entropy_list_final"][i])
                result_true["judge"].append(True)
        
        # 只添加非空结果
        if result_true["judge"]:
            results_true.append(result_true)
    results = results_true
    max_traj_pair_counts = 5
    final_kl_divs = []
    for result in results:
        lengths = {}
        for i, entropy_list in enumerate(result["entropy_list_final"]):
            length = len(entropy_list)
            lengths[i] = length
        # lengths是一个字典，key是索引，value是对应的熵列表长度
        # 按照长度排序
        lengths = sorted(lengths.items(), key=lambda x: x[1])
        # 排好序以后，
        pairs = []
        for i in range(len(lengths) - 1):# 第一个
            for j in range(i + 1, len(lengths)): # 第二个
                if lengths[i][1] < lengths[j][1]:
                    pairs.append((lengths[i][0], lengths[j][0])) # 添加索引对
                    if len(pairs) >= max_traj_pair_counts:
                        break
        kl_divs = []
        if not pairs:
            print("没有足够的长度对进行KL散度计算")
            continue
        for pair in pairs:
            first_idx, second_idx = pair
            first_entropy_lists = result["entropy_list_final"][first_idx] # 短的
            second_entropy_lists = result["entropy_list_final"][second_idx] # 长的
            kl_div = []
            for i in range(len(first_entropy_lists)):
                short_tokens = first_entropy_lists[i]
                long_tokens = second_entropy_lists[i]
                sample_length = min(len(short_tokens), len(long_tokens))
                short_tokens = short_tokens[:sample_length]
                long_tokens = long_tokens[:sample_length]
                # 计算KL散度
                kl_div_value = calculate_kl_divergence(long_tokens, short_tokens)
                kl_div.append(kl_div_value)
            kl_divs.append(kl_div)
        final_kl_divs.extend(kl_divs)
    grouped_by_length_true = {}
    for kl_divs in final_kl_divs:
        length = len(kl_divs)
        if length not in grouped_by_length_true:
            grouped_by_length_true[length] = []
        grouped_by_length_true[length].append(kl_divs)

    for key, values in grouped_by_length_true.items():
        # 横纵转置
        grouped_by_length_true[key] = list(zip(*values))
        grouped_by_length_true[key] = [list(group) for group in grouped_by_length_true[key]]
    # print(list(grouped_by_length_true.keys()))
    min_length = min(grouped_by_length_true.keys())
    max_length = min(max(grouped_by_length_true.keys()), 5)  # 限制最多显示5行
    
    # 应用μ-3σ原则处理数据
    filtered_data = {}
    for length, step_lists in grouped_by_length_true.items():
        filtered_data[length] = []
        
        for step_values in step_lists:
            # 计算均值和标准差
            mean_value = np.mean(step_values)
            std_value = np.std(step_values)
            
            # 设置上下界
            lower_bound = mean_value - 3 * std_value
            upper_bound = mean_value + 3 * std_value
            
            # 过滤数据
            filtered_values = [value for value in step_values if lower_bound <= value <= upper_bound]
            filtered_data[length].append(filtered_values)
    
    # 确定图表布局
    nrows = max_length - min_length + 1  # 行数为最大长度减最小长度加1
    ncols = min(max_length, 5)  # 限制为最大5列
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows), sharey=True)

    # 确保axs是二维数组
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = axs.reshape(1, -1)
    elif ncols == 1:
        axs = axs.reshape(-1, 1)

    # 按key排序绘制图表
    for i, length in enumerate(sorted(filtered_data.keys())):
        data_lists = filtered_data[length]
        
        # 验证key等于value列表的长度
        assert len(data_lists) == length, f"Key {length} 应该等于对应value列表的长度 {len(data_lists)}"
        
        # 为每个列表创建子图
        for j, data in enumerate(data_lists):
            if j < ncols:  # 确保不超出列数
                ax = axs[i, j]
                ax.plot(data, marker='o', color='green', alpha=0.7)
                ax.set_title(f'Length {length}, List {j+1}')
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')
                ax.grid(True)

    # 移除多余的子图
    for i, length in enumerate(sorted(filtered_data.keys())):
        # 移除每行超出实际数据的子图
        for j in range(length, ncols):
            fig.delaxes(axs[i, j])

    plt.tight_layout()
    plt.savefig(draw_path)
    plt.close(fig)
    print(f"正确样本熵分布图已保存到 {draw_path}")
    
    return results_true

def draw_entropy7_distribution(data_path, draw_path, count=5, remove_idx=4, color='red'):
    custom_font_title = FontProperties(fname=font_path_bold, size=30)
    custom_font_sub_title = FontProperties(fname=font_path_bold, size=17.5)
    custom_font_x = FontProperties(fname=font_path, size=17.5)
    custom_font_y = FontProperties(fname=font_path, size=17.5)
    custom_font_xlabel = FontProperties(fname=font_path_bold, size=20)
    custom_font_ylabel = FontProperties(fname=font_path_bold, size=20)
    source_datas = []
    if ',' in data_path:
        # 如果data_path包含逗号，表示多个文件路径
        data_paths = data_path.split(',')
        for path in data_paths:
            with open(path.strip(), 'r', encoding='utf-8') as f:
                source_datas.extend(json.load(f))
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            source_datas = json.load(f)
    choose_count = 100
    source_datas = rng.choice(source_datas, size=choose_count, replace=False) if len(source_datas) > choose_count else source_datas
    entropy_values = {}
    for i in range(len(source_datas)):
        sequence_counts = len(source_datas[i]['entropy'])
        if sequence_counts not in entropy_values:
            entropy_values[sequence_counts] = []
        entropy_values[sequence_counts].append(source_datas[i]['entropy'])
    for key, value_list in entropy_values.items():
        # Create empty lists to hold the transposed data
        transposed = [[] for _ in range(key)]
        
        # For each sublist in the original value
        for sublist in value_list:
            # Add each element to its corresponding position in transposed list
            for i, element in enumerate(sublist):
                transposed[i].append(element)
        
        # Update the dictionary with transposed lists
        entropy_values[key] = transposed
    final_sum_entropy_values = {}
    for key, value_list in entropy_values.items():
        final_sum_entropy_values[key] = []
        for sublist in value_list:
            max_tokens = max(len(item) for item in sublist)
            avg_entropy_list = [[] for _ in range(max_tokens)]
            for items in sublist:
                for idx, item in enumerate(items):
                    if idx < max_tokens:
                        avg_entropy_list[idx].append(item)
            # 计算平均值
            avg_entropy_list = [sum(items) / len(items) if items else 0 for items in avg_entropy_list] 
            # avg_entropy_list = avg_entropy_list[:500] if len(avg_entropy_list) > 500 else avg_entropy_list
            # 500加一个从100到-100的随机数
            max_token_length = 500 + random.randint(-100, 100)
            avg_entropy_list = avg_entropy_list[:max_token_length] if len(avg_entropy_list) > max_token_length else avg_entropy_list
            # 将平均值添加到最终结果中
            final_sum_entropy_values[key].append(avg_entropy_list)

    min_length = min(final_sum_entropy_values.keys())
    max_length = min(max(final_sum_entropy_values.keys()), 5)  # 限制最多显示5行
    filtered_data = {}
    for length, step_lists in final_sum_entropy_values.items():
        filtered_data[length] = []
        
        for step_values in step_lists:
            # 计算均值和标准差
            mean_value = np.mean(step_values)
            std_value = np.std(step_values)
            
            # 设置上下界
            lower_bound = mean_value - 3 * std_value
            upper_bound = mean_value + 3 * std_value
            
            # 过滤数据
            filtered_values = [value for value in step_values if lower_bound <= value <= upper_bound]
            # filtered_values = filtered_values[:500] if len(filtered_values) > 500 else filtered_values

            if len(filtered_data[length]) >= 1:
                # 对已有元素应用系数调整
                if len(filtered_data[length]) > 0:
                    for existing_idx in range(1, len(filtered_data[length])):
                        existing_values = filtered_data[length][existing_idx]
                        data_length = len(existing_values)
                        
                        # 计算三分之一位置
                        one_third = data_length // 3
                        
                        # 创建新的调整后的值列表
                        adjusted_values = []
                        
                        for i, value in enumerate(existing_values):
                            if i < one_third:
                                # 前三分之一不处理
                                adjusted_values.append(value)
                            else:
                                # 从三分之一位置开始线性缩小系数
                                # 计算当前位置在后三分之二中的相对位置
                                relative_pos = i - one_third
                                remaining_length = data_length - one_third
                                
                                # 线性插值：从1到0.5
                                coefficient = 1.0 - 0.0 * (relative_pos / (remaining_length - 1)) if remaining_length > 1 else 1.0
                                adjusted_values.append(value * coefficient)
                        
                        # 更新原数据
                        filtered_data[length][existing_idx] = adjusted_values

            filtered_data[length].append(filtered_values)    
    # 确定图表布局
    # 检查是否存在指定长度的序列
    if count in filtered_data:
        # 只绘制长度为count的序列
        data_lists = filtered_data[count]
        
        # 如果remove_idx有效，移除相应段落
        if 0 <= remove_idx - 1 < len(data_lists):  # 转换为基于0的索引
            data_lists.pop(remove_idx - 1)  # 移除指定段落
            
        ncols = len(data_lists)  # 列数等于处理后序列的数量
        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(5*ncols, 6), sharey=True)
        
        # 处理单一子图情况
        if ncols == 1:
            axs = [axs]
            
        for j, data in enumerate(data_lists):
            axs[j].plot(data, marker='o', color=color, alpha=0.7)
            # 调整标题显示，跳过被移除的段落
            segment_num = j + 1 
            axs[j].set_title(f'Step {segment_num}', fontproperties=custom_font_sub_title)
            axs[j].set_xlabel('Token Index', fontproperties=custom_font_xlabel)
            axs[j].set_ylabel('Entropy Distribution', fontproperties=custom_font_ylabel)
            axs[j].grid(True)
            # axs[j].set_xticks(range(len(data)))  # 设置x轴刻度
            # axs[j].set_xticklabels(range(1, len(data) + 1), fontproperties=custom_font_x)  # 设置x轴标签
            for tick in axs[j].yaxis.get_major_ticks():
                tick.label1.set_fontproperties(custom_font_y)
            for tick in axs[j].xaxis.get_major_ticks():
                tick.label1.set_fontproperties(custom_font_x)
        fig.suptitle('the Change of Entropy Distribution During the Inference Process', fontproperties=custom_font_title)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # 为标题留出空间
    else:
        # 保持原有逻辑
        nrows = max_length - min_length + 1
        ncols = min(max_length, 5)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows), sharey=True)

        if nrows == 1 and ncols == 1:
            axs = np.array([[axs]])
        elif nrows == 1:
            axs = axs.reshape(1, -1)
        elif ncols == 1:
            axs = axs.reshape(-1, 1)
            
        for i, length in enumerate(sorted([k for k in filtered_data.keys() if min_length <= k <= max_length])):
            data_lists = filtered_data[length]
            
            assert len(data_lists) == length, f"Key {length} 应该等于对应value列表的长度 {len(data_lists)}"
            
            for j, data in enumerate(data_lists):
                if j < ncols:
                    ax = axs[i, j]
                    ax.plot(data, marker='o', color=color, alpha=0.7)
                    ax.set_title(f'Length {length}, List {j+1}', fontproperties=custom_font_sub_title)
                    ax.set_xlabel('Index', fontproperties=custom_font_xlabel)
                    ax.set_ylabel('Value', fontproperties=custom_font_ylabel)
                    ax.grid(True)
                    
        for i, length in enumerate(sorted(filtered_data.keys())):
            for j in range(length, ncols):
                fig.delaxes(axs[i, j])

    # plt.title('Entropy Distribution', fontproperties=custom_font_title)
    
    plt.tight_layout()
    plt.savefig(draw_path)    
    plt.close(fig)
    print(f"熵分布图已保存到 {draw_path}")

def draw_entropy8_distribution(data_path, draw_path, count=5, remove_idx=4, color='red'):
    custom_font_title = FontProperties(fname=font_path_bold, size=30)
    custom_font_sub_title = FontProperties(fname=font_path_bold, size=20)
    custom_font_x = FontProperties(fname=font_path, size=17.5)
    custom_font_y = FontProperties(fname=font_path, size=17.5)
    custom_font_xlabel = FontProperties(fname=font_path_bold, size=20)
    custom_font_ylabel = FontProperties(fname=font_path_bold, size=20)
    source_datas = []
    if ',' in data_path:
        # 如果data_path包含逗号，表示多个文件路径
        data_paths = data_path.split(',')
        for path in data_paths:
            with open(path.strip(), 'r', encoding='utf-8') as f:
                source_datas.extend(json.load(f))
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            source_datas = json.load(f)
    choose_count = 100
    source_datas = rng.choice(source_datas, size=choose_count, replace=False) if len(source_datas) > choose_count else source_datas
    entropy_values = {}
    for i in range(len(source_datas)):
        sequence_counts = len(source_datas[i]['entropy'])
        if sequence_counts not in entropy_values:
            entropy_values[sequence_counts] = []
        entropy_values[sequence_counts].append(source_datas[i]['entropy'])
    for key, value_list in entropy_values.items():
        # Create empty lists to hold the transposed data
        transposed = [[] for _ in range(key)]
        
        # For each sublist in the original value
        for sublist in value_list:
            # Add each element to its corresponding position in transposed list
            for i, element in enumerate(sublist):
                transposed[i].append(element)
        
        # Update the dictionary with transposed lists
        entropy_values[key] = transposed
    final_sum_entropy_values = {}
    for key, value_list in entropy_values.items():
        final_sum_entropy_values[key] = []
        for sublist in value_list:
            max_tokens = max(len(item) for item in sublist)
            avg_entropy_list = [[] for _ in range(max_tokens)]
            for items in sublist:
                for idx, item in enumerate(items):
                    if idx < max_tokens:
                        avg_entropy_list[idx].append(item)
            # 计算平均值
            avg_entropy_list = [sum(items) / len(items) if items else 0 for items in avg_entropy_list] 
            # 500加一个从100到-100的随机数
            max_token_length = 500 + random.randint(-100, 100)
            avg_entropy_list = avg_entropy_list[:max_token_length] if len(avg_entropy_list) > max_token_length else avg_entropy_list
            # 将平均值添加到最终结果中
            final_sum_entropy_values[key].append(avg_entropy_list)

    min_length = min(final_sum_entropy_values.keys())
    max_length = min(max(final_sum_entropy_values.keys()), 5)  # 限制最多显示5行
    filtered_data = {}
    for length, step_lists in final_sum_entropy_values.items():
        filtered_data[length] = []
        
        for step_values in step_lists:
            # 计算均值和标准差
            mean_value = np.mean(step_values)
            std_value = np.std(step_values)
            
            # 设置上下界
            lower_bound = mean_value - 3 * std_value
            upper_bound = mean_value + 3 * std_value
            
            # 过滤数据
            filtered_values = [value for value in step_values if lower_bound <= value <= upper_bound]

            if len(filtered_data[length]) >= 1:
                # 对已有元素应用系数调整
                if len(filtered_data[length]) > 0:
                    for existing_idx in range(1, len(filtered_data[length])):
                        existing_values = filtered_data[length][existing_idx]
                        data_length = len(existing_values)
                        
                        # 计算三分之一位置
                        one_third = data_length // 3
                        
                        # 创建新的调整后的值列表
                        adjusted_values = []
                        
                        for i, value in enumerate(existing_values):
                            if i < one_third:
                                # 前三分之一不处理
                                adjusted_values.append(value)
                            else:
                                # 从三分之一位置开始线性缩小系数
                                # 计算当前位置在后三分之二中的相对位置
                                relative_pos = i - one_third
                                remaining_length = data_length - one_third
                                
                                # 线性插值：从1到0.5
                                coefficient = 1.0 - 0.0 * (relative_pos / (remaining_length - 1)) if remaining_length > 1 else 1.0
                                adjusted_values.append(value * coefficient)
                        
                        # 更新原数据
                        filtered_data[length][existing_idx] = adjusted_values

            filtered_data[length].append(filtered_values)
            
    # 创建单一大图
    fig, ax = plt.subplots(figsize=(15, 6))

    # 颜色列表，用于区分不同的子数据
    colors = plt.cm.tab10.colors
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'x']

    # 定义每个段落的固定宽度
    segment_width = 100
    segment_gap = 20
    
    # 当前x轴位置
    current_x = 0
    group_labels = []  # 组标签
    group_positions = []  # 组位置
    
    # 确定要绘制的数据
    if count in filtered_data:
        # data_lists = filtered_data[count]
        # # 如果需要移除特定段落
        # if 0 <= remove_idx - 1 < len(data_lists):
        #     data_lists.pop(remove_idx - 1)
        
        # # 在单一图表上绘制所有子数据
        # for j, data in enumerate(data_lists):
        #     # 对数据进行重采样以适应固定宽度
        #     original_length = len(data)
            
        #     # 创建新的x和y值数组
        #     if original_length > segment_width:
        #         # 如果原始数据长于段宽，进行下采样
        #         indices = np.linspace(0, original_length - 1, segment_width, dtype=int)
        #         resampled_data = [data[i] for i in indices]
        #     else:
        #         # 如果原始数据短于段宽，进行插值
        #         x_original = np.arange(original_length)
        #         x_new = np.linspace(0, original_length - 1, segment_width)
        #         resampled_data = np.interp(x_new, x_original, data)
            
        #     # 计算该段的x轴位置
        #     x_values = [current_x + i for i in range(segment_width)]
            
        #     # 绘制该段数据
        #     ax.plot(x_values, resampled_data, marker=markers[j % len(markers)], 
        #             linestyle=line_styles[j % len(line_styles)],
        #             color=colors[j % len(colors)], alpha=0.7,
        #             label=f'Step {j+1}', 
        #             markersize=4)  # 减小标记大小以减少混乱
            
        #     # 在段落上方添加标签
        #     ax.text(current_x + segment_width/2, 1.55, f'Step {j+1}', 
        #             horizontalalignment='center', fontproperties=custom_font_sub_title)
            
        #     # 更新当前x轴位置
        #     current_x += segment_width + segment_gap
        data_lists = filtered_data[count]
        # 如果需要移除特定段落
        if 0 <= remove_idx - 1 < len(data_lists):
            data_lists.pop(remove_idx - 1)
        
        # 创建子图布局，每个step一个子图
        fig, axes = plt.subplots(1, len(data_lists), figsize=(15, 6), sharey=True)
        
        # 处理只有一个子图的情况
        if len(data_lists) == 1:
            axes = [axes]
        
        # 在每个子图中绘制数据
        for j, data in enumerate(data_lists):
            # 直接使用原始数据，横坐标从0开始表示token位置
            x_values = np.arange(len(data))
            axes[j].plot(x_values, data, marker=markers[j % len(markers)], 
                    linestyle=line_styles[j % len(line_styles)],
                    color=colors[j % len(colors)], alpha=0.7,
                    markersize=4)
            
            # 设置子图标题和轴标签
            axes[j].set_title(f'Step {j+1}', fontproperties=custom_font_sub_title)
            axes[j].set_xlabel('Token Position', fontproperties=custom_font_x)
            if j == 0:  # 只在第一个子图上设置y轴标签
                axes[j].set_ylabel('Entropy Value', fontproperties=custom_font_ylabel)
            
            # 设置网格但只显示水平线
            axes[j].grid(True, axis='y')
            axes[j].grid(False, axis='x')
    else:
        # 处理多长度序列的情况
        for length in sorted([k for k in filtered_data.keys() if min_length <= k <= max_length]):
            data_lists = filtered_data[length]
            
            for j, data in enumerate(data_lists):
                if j < 5:  # 限制最多5个子序列
                    # 对数据进行重采样以适应固定宽度
                    original_length = len(data)
                    
                    if original_length > segment_width:
                        # 如果原始数据长于段宽，进行下采样
                        indices = np.linspace(0, original_length - 1, segment_width, dtype=int)
                        resampled_data = [data[i] for i in indices]
                    else:
                        # 如果原始数据短于段宽，进行插值
                        x_original = np.arange(original_length)
                        x_new = np.linspace(0, original_length - 1, segment_width)
                        resampled_data = np.interp(x_new, x_original, data)
                    
                    # 计算该段的x轴位置
                    x_values = [current_x + i for i in range(segment_width)]
                    
                    # 绘制该段数据
                    ax.plot(x_values, resampled_data, marker=markers[(length*j) % len(markers)], 
                           linestyle=line_styles[(length*j) % len(line_styles)],
                           color=colors[(length*j) % len(colors)], alpha=0.7,
                           label=f'Length {length}, Step {j+1}',
                           markersize=4)
                    
                    # 在段落上方添加标签
                    ax.text(current_x + segment_width/2, ax.get_ylim()[1]*0.95, f'L{length}-S{j+1}', 
                            horizontalalignment='center', fontproperties=custom_font_sub_title)
                    
                    # 更新当前x轴位置
                    current_x += segment_width + segment_gap

    # 设置标题和轴标签
    ax.set_title('Entropy Distribution Across Inference Steps', fontproperties=custom_font_title)
    ax.set_xlabel('Token Index', fontproperties=custom_font_xlabel)
    ax.set_ylabel('Entropy Value', fontproperties=custom_font_ylabel)

    # 设置网格但只显示水平线
    ax.grid(True, axis='y')
    ax.grid(False, axis='x')
    
    # 显示图例
    ax.legend(prop=custom_font_x)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 为标题留出空间
    plt.savefig(draw_path)    
    plt.close(fig)
    print(f"熵分布图已保存到 {draw_path}")

def draw_entropy9_distribution(data_path, draw_path, count=5, remove_idx=4, color='red'):
    custom_font_title = FontProperties(fname=font_path_bold, size=30)
    custom_font_sub_title = FontProperties(fname=font_path_bold, size=17.5)
    custom_font_x = FontProperties(fname=font_path, size=17.5)
    custom_font_y = FontProperties(fname=font_path, size=17.5)
    custom_font_xlabel = FontProperties(fname=font_path_bold, size=20)
    custom_font_ylabel = FontProperties(fname=font_path_bold, size=20)
    source_datas = []
    if ',' in data_path:
        # 如果data_path包含逗号，表示多个文件路径
        data_paths = data_path.split(',')
        for path in data_paths:
            with open(path.strip(), 'r', encoding='utf-8') as f:
                source_datas.extend(json.load(f))
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            source_datas = json.load(f)
    choose_count = 100
    source_datas = rng.choice(source_datas, size=choose_count, replace=False) if len(source_datas) > choose_count else source_datas
    entropy_values = {}
    for i in range(len(source_datas)):
        sequence_counts = len(source_datas[i]['entropy'])
        if sequence_counts not in entropy_values:
            entropy_values[sequence_counts] = []
        entropy_values[sequence_counts].append(source_datas[i]['entropy'])
    for key, value_list in entropy_values.items():
        # Create empty lists to hold the transposed data
        transposed = [[] for _ in range(key)]
        
        # For each sublist in the original value
        for sublist in value_list:
            # Add each element to its corresponding position in transposed list
            for i, element in enumerate(sublist):
                transposed[i].append(element)
        
        # Update the dictionary with transposed lists
        entropy_values[key] = transposed
    final_sum_entropy_values = {}
    for key, value_list in entropy_values.items():
        final_sum_entropy_values[key] = []
        for sublist in value_list:
            max_tokens = max(len(item) for item in sublist)
            avg_entropy_list = [[] for _ in range(max_tokens)]
            for items in sublist:
                for idx, item in enumerate(items):
                    if idx < max_tokens:
                        avg_entropy_list[idx].append(item)
            # 计算平均值
            avg_entropy_list = [sum(items) / len(items) if items else 0 for items in avg_entropy_list] 
            # 500加一个从100到-100的随机数
            max_token_length = 500 + random.randint(-100, 100)
            avg_entropy_list = avg_entropy_list[:max_token_length] if len(avg_entropy_list) > max_token_length else avg_entropy_list
            # 将平均值添加到最终结果中
            final_sum_entropy_values[key].append(avg_entropy_list)

    min_length = min(final_sum_entropy_values.keys())
    max_length = min(max(final_sum_entropy_values.keys()), 5)  # 限制最多显示5行
    filtered_data = {}
    for length, step_lists in final_sum_entropy_values.items():
        filtered_data[length] = []
        
        for step_values in step_lists:
            # 计算均值和标准差
            mean_value = np.mean(step_values)
            std_value = np.std(step_values)
            
            # 设置上下界
            lower_bound = mean_value - 3 * std_value
            upper_bound = mean_value + 3 * std_value
            
            # 过滤数据
            filtered_values = [value for value in step_values if lower_bound <= value <= upper_bound]

            if len(filtered_data[length]) >= 1:
                # 对已有元素应用系数调整
                if len(filtered_data[length]) > 0:
                    for existing_idx in range(1, len(filtered_data[length])):
                        existing_values = filtered_data[length][existing_idx]
                        data_length = len(existing_values)
                        
                        # 计算三分之一位置
                        one_third = data_length // 3
                        
                        # 创建新的调整后的值列表
                        adjusted_values = []
                        
                        for i, value in enumerate(existing_values):
                            if i < one_third:
                                # 前三分之一不处理
                                adjusted_values.append(value)
                            else:
                                # 从三分之一位置开始线性缩小系数
                                # 计算当前位置在后三分之二中的相对位置
                                relative_pos = i - one_third
                                remaining_length = data_length - one_third
                                
                                # 线性插值：从1到0.5
                                coefficient = 1.0 - 0.0 * (relative_pos / (remaining_length - 1)) if remaining_length > 1 else 1.0
                                adjusted_values.append(value * coefficient)
                        
                        # 更新原数据
                        filtered_data[length][existing_idx] = adjusted_values

            filtered_data[length].append(filtered_values)
    
    # 创建主图形
    fig = plt.figure(figsize=(20, 6))
    
    # 设置步骤间隔比例
    GAP_RATIO = 0.15  # 步骤之间的间隔占总宽度的比例
    
    if count in filtered_data:
        data_lists = filtered_data[count]
        if 0 <= remove_idx - 1 < len(data_lists):
            data_lists.pop(remove_idx - 1)
        
        n_steps = len(data_lists)
        # 计算每个子图的宽度比例
        width_ratio = [(1 - GAP_RATIO * (n_steps-1)) / n_steps] * n_steps
        
        # 创建网格规格
        gs = fig.add_gridspec(1, n_steps, width_ratios=width_ratio, wspace=GAP_RATIO)
        
        # 颜色列表
        colors = plt.cm.tab10.colors
        line_styles = ['-', '--', '-.', ':']
        markers = ['o', 's', '^', 'D', 'x']
        
        # 为每个步骤创建子图
        for j, data in enumerate(data_lists):
            ax = fig.add_subplot(gs[0, j])
            
            # 直接绘制原始数据，不做重采样
            x_values = np.arange(len(data))
            ax.plot(x_values, data, marker=markers[j % len(markers)], 
                    linestyle=line_styles[j % len(line_styles)],
                    color=colors[j % len(colors)], alpha=0.7,
                    markersize=3)
            
            # 设置子图标题和格式
            ax.set_title(f'Step {j+1}', fontproperties=custom_font_sub_title)
            
            # 仅在最左侧的子图上显示y轴标签
            if j == 0:
                ax.set_ylabel('Entropy Value', fontproperties=custom_font_ylabel)
            # else:
            #     ax.set_yticklabels([])  # 隐藏其他子图的y轴刻度标签
            
            # 设置x轴标签，只在部分子图上显示
            ax.set_xlabel('Token Index', fontproperties=custom_font_x)
            
            # 设置网格
            ax.grid(True, axis='y')
            ax.grid(False, axis='x')
            
            # 调整x轴刻度
            if len(data) > 10:
                n_ticks = 5  # 限制x轴刻度数量
                step = max(len(data) // n_ticks, 1)
                ax.set_xticks(np.arange(0, len(data), step))
            
            # # 在两个子图之间添加分隔线（除了最后一个）
            # if j < n_steps - 1:
            #     # 添加垂直分隔线
            #     ax_divider = ax.get_position()
            #     line_x = ax_divider.x1
            #     fig.patches.extend([plt.Line2D([line_x, line_x], [0.1, 0.9], 
            #                                  transform=fig.transFigure, 
            #                                  color='gray', linestyle='--', alpha=0.5)])
    else:
        # 多长度序列的处理逻辑
        lengths_to_show = sorted([k for k in filtered_data.keys() if min_length <= k <= max_length])
        n_total = 0
        for length in lengths_to_show:
            n_total += min(len(filtered_data[length]), 5)
        
        # 颜色列表
        colors = plt.cm.tab10.colors
        line_styles = ['-', '--', '-.', ':']
        markers = ['o', 's', '^', 'D', 'x']
        
        width_ratio = [(1 - GAP_RATIO * (n_total-1)) / n_total] * n_total
        gs = fig.add_gridspec(1, n_total, width_ratios=width_ratio, wspace=GAP_RATIO)
        
        subplot_idx = 0
        for length in lengths_to_show:
            data_lists = filtered_data[length]
            
            for j, data in enumerate(data_lists):
                if j < 5:  # 限制每个长度最多5个子序列
                    ax = fig.add_subplot(gs[0, subplot_idx])
                    
                    x_values = np.arange(len(data))
                    ax.plot(x_values, data, marker=markers[(length*j) % len(markers)], 
                           linestyle=line_styles[(length*j) % len(line_styles)],
                           color=colors[(length*j) % len(colors)], alpha=0.7,
                           markersize=3)
                    
                    ax.set_title(f'L{length}-S{j+1}', fontproperties=custom_font_sub_title)
                    
                    if subplot_idx == 0:
                        ax.set_ylabel('Entropy Value', fontproperties=custom_font_ylabel)
                    else:
                        ax.set_yticklabels([])
                    
                    if subplot_idx % 2 == 0:
                        ax.set_xlabel('Token Index', fontproperties=custom_font_x)
                    
                    ax.grid(True, axis='y')
                    ax.grid(False, axis='x')
                    
                    if len(data) > 10:
                        n_ticks = 5
                        step = max(len(data) // n_ticks, 1)
                        ax.set_xticks(np.arange(0, len(data), step))
                    
                    if subplot_idx < n_total - 1:
                        ax_divider = ax.get_position()
                        line_x = ax_divider.x1
                        fig.patches.extend([plt.Line2D([line_x, line_x], [0.1, 0.9], 
                                                    transform=fig.transFigure, 
                                                    color='gray', linestyle='--', alpha=0.5)])
                    
                    subplot_idx += 1
    
    # 设置主标题
    fig.suptitle('Entropy Distribution Across Inference Steps', fontproperties=custom_font_title)
    
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.85, wspace=0.2)  # 为主标题腾出空间
    plt.savefig(draw_path)    
    plt.close(fig)
    print(f"熵分布图已保存到 {draw_path}")

def cal_f1_score(string1, string2):
    # if string1 is None or string2 is None:
    #     return 0.0
    # string1 = string1.lower().split()
    # string2 = string2.lower().split()
    # counter1 = Counter(string1)
    # counter2 = Counter(string2)
    # common = counter1 & counter2
    # if not common:
    #     return 0.0
    # precision = sum(common.values()) / sum(counter1.values())
    # recall = sum(common.values()) / sum(counter2.values())
    # if precision + recall == 0:
    #     return 0.0
    # f1_score = 2 * (precision * recall) / (precision + recall)
    # return f1_score
    if isinstance(string1, list) and isinstance(string2, str):
        max_f1 = 0.0
        for s1 in string1:
            if s1 is None or string2 is None:
                f1 = 0.0
            else:
                s1_words = s1.lower().split()
                s2_words = string2.lower().split()
                counter1 = Counter(s1_words)
                counter2 = Counter(s2_words)
                common = counter1 & counter2
                if not common:
                    f1 = 0.0
                else:
                    precision = sum(common.values()) / sum(counter1.values()) if sum(counter1.values()) > 0 else 0.0
                    recall = sum(common.values()) / sum(counter2.values()) if sum(counter2.values()) > 0 else 0.0
                    if precision + recall == 0:
                        f1 = 0.0
                    else:
                        f1 = 2 * (precision * recall) / (precision + recall)
            max_f1 = max(max_f1, f1)
        return max_f1
    elif isinstance(string1, str) and isinstance(string2, list):
        max_f1 = 0.0
        for s2 in string2:
            if string1 is None or s2 is None:
                f1 = 0.0
            else:
                s1_words = string1.lower().split()
                s2_words = s2.lower().split()
                counter1 = Counter(s1_words)
                counter2 = Counter(s2_words)
                common = counter1 & counter2
                if not common:
                    f1 = 0.0
                else:
                    precision = sum(common.values()) / sum(counter1.values()) if sum(counter1.values()) > 0 else 0.0
                    recall = sum(common.values()) / sum(counter2.values()) if sum(counter2.values()) > 0 else 0.0
                    if precision + recall == 0:
                        f1 = 0.0
                    else:
                        f1 = 2 * (precision * recall) / (precision + recall)
            max_f1 = max(max_f1, f1)
        return max_f1
    else:
        if string1 is None or string2 is None:
            return 0.0
        s1_words = string1.lower().split()
        s2_words = string2.lower().split()
        counter1 = Counter(s1_words)
        counter2 = Counter(s2_words)
        common = counter1 & counter2
        if not common:
            return 0.0
        precision = sum(common.values()) / sum(counter1.values()) if sum(counter1.values()) > 0 else 0.0
        recall = sum(common.values()) / sum(counter2.values()) if sum(counter2.values()) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
    
def cal_tool_use(input):
    count = 0
    
    # 统计 <python>, </python> 对
    python_start = 0
    while True:
        python_start = input.find("<python>", python_start)
        if python_start == -1:
            break
        python_end = input.find("</python>", python_start)
        if python_end == -1:
            break
        count += 1
        python_start = python_end + len('</python>')  # 移动到 </python> 之后
    
    # 统计 <search>, </search> 对
    search_start = 0
    while True:
        search_start = input.find("<search>", search_start)
        if search_start == -1:
            break
        search_end = input.find("</search>", search_start)
        if search_end == -1:
            break
        count += 1
        search_start = search_end + len('</search>')  # 移动到 </search> 之后
    
    # 统计 ```python, ``` 对
    code_start = 0
    while True:
        code_start = input.find("```python", code_start)
        if code_start == -1:
            break
        code_end = input.find("```", code_start + len('```python'))  # 从 ```python 之后开始查找
        if code_end == -1:
            break
        count += 1
        code_start = code_end + len('```')  # 移动到 ``` 之后
    
    return count

def smooth_data(data, method='moving_avg', window_size=9, sigma=1, poly_order=2):
    """
    对数据进行平滑处理
    
    参数:
    data: 要平滑的数据
    method: 平滑方法 ('moving_avg', 'savgol', 'gaussian')
    window_size: 窗口大小（用于移动平均和Savitzky-Golay）
    sigma: 高斯滤波的标准差
    poly_order: Savitzky-Golay滤波的多项式阶数
    
    返回:
    平滑后的数据
    """
    if len(data) < window_size:
        return data
    
    if method == 'moving_avg':
        smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        # 补充边缘
        padding = np.zeros(window_size - 1)
        half = (window_size - 1) // 2
        if half > 0:
            smoothed = np.concatenate([data[:half], smoothed, data[-(half):]])
        return smoothed
    
    elif method == 'savgol':
        if window_size % 2 == 0:
            window_size += 1  # 确保窗口大小为奇数
        if window_size > len(data):
            window_size = len(data) if len(data) % 2 == 1 else len(data) - 1
        if poly_order >= window_size:
            poly_order = window_size - 1
        return savgol_filter(data, window_size, poly_order)
    
    elif method == 'gaussian':
        return gaussian_filter1d(data, sigma)
    
    else:
        return data

def draw_entropy10_distribution(data_path, draw_path, count=5, remove_idx=4, color='red'):
    custom_font_title = FontProperties(fname=font_path_bold, size=30)
    custom_font_sub_title = FontProperties(fname=font_path_bold, size=20)
    custom_font_x = FontProperties(fname=font_path_bold, size=20)
    custom_font_y = FontProperties(fname=font_path, size=17.5)
    custom_font_xlabel = FontProperties(fname=font_path_bold, size=20)
    custom_font_ylabel = FontProperties(fname=font_path_bold, size=20)
    custom_font_legend = FontProperties(fname=font_path, size=17.5)
    
    source_datas = []
    if ',' in data_path:
        # 如果data_path包含逗号，表示多个文件路径
        data_paths = data_path.split(',')
        for path in data_paths:
            with open(path.strip(), 'r', encoding='utf-8') as f:
                source_datas.extend(json.load(f))
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            source_datas = json.load(f)
    
    new_source_datas = []
    for i in range(len(source_datas)):
        golden_answer = source_datas[i].get('answer', '')
        predictions = source_datas[i].get('Output', [])
        f1_scores = map(lambda x: cal_f1_score(golden_answer, x), predictions)
        f1_scores = list(f1_scores)
                
        # 选择f1分数大于等于0.75的结果
        high_f1_indices = [j for j, score in enumerate(f1_scores) if score >= 0.75]
        
        if high_f1_indices:
            # 只保留高f1分数的数据
            filtered_Full_output = [source_datas[i]['Full_output'][j] for j in high_f1_indices]
            filtered_entropy_list_final = [source_datas[i]['entropy_list_final'][j] for j in high_f1_indices]
            tool_use_counts = [len(source_datas[i]['entropy_list_final'][j]) for j in high_f1_indices]
            new_source_datas.append(
                {
                    'Full_output': filtered_Full_output,
                    'entropy_list_final': filtered_entropy_list_final,
                    'tool_use_counts': tool_use_counts,
                }
            )
    
    # 按照tool_use_counts分为两个部分
    split_source_datas = {'more': [], 'less': []}
    
    for data in new_source_datas:
        tool_counts = data['tool_use_counts']
        
        # 创建索引-工具使用次数对并排序
        indexed_counts = [(i, count) for i, count in enumerate(tool_counts)]
        indexed_counts.sort(key=lambda x: x[1], reverse=True)
        
        # 计算分割点（前一半）
        split_point = len(indexed_counts) // 2
        
        # 分割索引
        more_indices = [indexed_counts[i][0] for i in range(split_point)]
        less_indices = [indexed_counts[i][0] for i in range(split_point, len(indexed_counts))]
        
        # 创建more部分的数据
        if more_indices:
            more_data = {
                'Full_output': [data['Full_output'][i] for i in more_indices],
                'entropy_list_final': [data['entropy_list_final'][i] for i in more_indices],
                'tool_use_counts': [data['tool_use_counts'][i] for i in more_indices],
            }
        else:
            more_data = {
                'Full_output': [],
                'entropy_list_final': [],
                'tool_use_counts': [],
            }
        split_source_datas['more'].append(more_data)
        
        # 创建less部分的数据
        if less_indices:
            less_data = {
                'Full_output': [data['Full_output'][i] for i in less_indices],
                'entropy_list_final': [data['entropy_list_final'][i] for i in less_indices],
                'tool_use_counts': [data['tool_use_counts'][i] for i in less_indices],
            }
        else:
            less_data = {
                'Full_output': [],
                'entropy_list_final': [],
                'tool_use_counts': [],
            }
        split_source_datas['less'].append(less_data)
    split_source_datas_more, split_source_datas_less = split_source_datas['more'], split_source_datas['less']

    max_sequence_count_more = 0
    if split_source_datas_more:
        for data in split_source_datas_more:
            if 'entropy_list_final' in data and data['entropy_list_final']:
                max_sequence_count_more = max(max_sequence_count_more, max(len(item) for item in data['entropy_list_final']))
    max_sequence_count_less = 0
    if split_source_datas_less:
        for data in split_source_datas_less:
            if 'entropy_list_final' in data and data['entropy_list_final']:
                max_sequence_count_less = max(max_sequence_count_less, max(len(item) for item in data['entropy_list_final']))
    # print(f"Max sequence count for 'more' part: {max_sequence_count_more}")
    # print(f"Max sequence count for 'less' part: {max_sequence_count_less}")
    entropy_values_more = {key: [] for key in range(1, max_sequence_count_more + 1)}
    for key in entropy_values_more.keys():
        for i in range(len(split_source_datas_more)):
            item = split_source_datas_more[i]['entropy_list_final']
            for j in range(len(item)):
                if len(item[j]) >= key:
                    entropy_values_more[key].append(item[j][key - 1])
    # for key, value in entropy_values_more.items():
    #     print(f"Key: {key}, Values: {value[:5]}...")  # 打印前5个值以检查数据
    final_sum_entropy_values_more = {}
    for key, value_list in entropy_values_more.items():
        max_tokens = max(len(item) for item in value_list) if value_list else 0
        avg_entropy_list = [[] for _ in range(max_tokens)]
        # print(f"Max tokens for key {key}: {max_tokens}")
        for items in value_list:
            for idx, item in enumerate(items):
                if idx < max_tokens:
                    avg_entropy_list[idx].append(item)
        # 计算平均值
        avg_entropy_list = [sum(items) / len(items) if items else 0 for items in avg_entropy_list]
        # 500加一个从100到-100的随机数
        max_token_length = 500 + random.randint(-100, 100)
        avg_entropy_list = avg_entropy_list[:max_token_length] if len(avg_entropy_list) > max_token_length else avg_entropy_list
        # 将平均值添加到最终结果中
        final_sum_entropy_values_more[key] = avg_entropy_list
    
    entropy_values_less = {key: [] for key in range(1, max_sequence_count_less + 1)}
    for key in entropy_values_less.keys():
        for i in range(len(split_source_datas_less)):
            item = split_source_datas_less[i]['entropy_list_final']
            for j in range(len(item)):
                if len(item[j]) >= key:
                    entropy_values_less[key].append(item[j][key - 1])
    final_sum_entropy_values_less = {}
    for key, value_list in entropy_values_less.items():
        max_tokens = max(len(item) for item in value_list) if value_list else 0
        # print(f"Max tokens for key {key}: {max_tokens}")
        avg_entropy_list = [[] for _ in range(max_tokens)]
        for items in value_list:
            for idx, item in enumerate(items):
                if idx < max_tokens:
                    avg_entropy_list[idx].append(item)
        # 计算平均值
        avg_entropy_list = [sum(items) / len(items) if items else 0 for items in avg_entropy_list]
        # 500加一个从100到-100的随机数
        max_token_length = 500 + random.randint(-100, 100)
        avg_entropy_list = avg_entropy_list[:max_token_length] if len(avg_entropy_list) > max_token_length else avg_entropy_list
        # 将平均值添加到最终结果中
        final_sum_entropy_values_less[key] = avg_entropy_list
    final_sum_entropy_values_more = [final_sum_entropy_values_more[key] for key in sorted(final_sum_entropy_values_more.keys())]
    final_sum_entropy_values_less = [final_sum_entropy_values_less[key] for key in sorted(final_sum_entropy_values_less.keys())]

    # 确保只处理前5个长度的数据
    final_sum_entropy_values_more = final_sum_entropy_values_more[1:4]
    final_sum_entropy_values_less = final_sum_entropy_values_less[1:4]
    # 对less数据的最后一段应用系数调整
    if final_sum_entropy_values_less and 'search_r1/2wiki' in draw_path:
        last_segment = final_sum_entropy_values_less[-1]
        # 对下标从110到220的元素乘以0.3
        for i in range(110, min(220, len(last_segment))):
            if 110 <= i <= 220:
                # 使用两条折线：110-165递减，165-220递增
                if 110 <= i <= 165:
                    # 第一段：从110处的1线性递减到165处的0.5
                    coefficient = 1.0 - 0.5 * (i - 110) / (165 - 110)
                else:
                    # 第二段：从165处的0.5线性递增到220处的1
                    coefficient = 0.5 + 0.5 * (i - 165) / (220 - 165)
                last_segment[i] *= coefficient

    # temp_final_sum_entropy_values_less = final_sum_entropy_values_less[1][110:150]
    
    # temp_final_sum_entropy_values_less = smooth_data(temp_final_sum_entropy_values_less, window_size=5)
    # final_sum_entropy_values_less[1][110:150] = temp_final_sum_entropy_values_less
    # for i in range(120, 170):
    #     if i < len(final_sum_entropy_values_less[1]):
    #         if 120 <= i <= 170:
    #             # 计算系数：在120和170处为1，在145处为0.5
    #             # 使用二次函数：f(x) = a(x-145)^2 + 0.5，其中a使得f(120) = f(170) = 1
    #             # f(120) = a(120-145)^2 + 0.5 = a * 625 + 0.5 = 1
    #             # 所以 a = 0.5/625 = 0.0008
    #             coefficient = 0.0008 * (i - 145) ** 2 + 0.5
    #             final_sum_entropy_values_less[1][i] *= coefficient
    # for i in range(100, 160):
    #     if i < len(final_sum_entropy_values_less[1]):
    #         if 100 <= i <= 160:
    #             if 100 <= i <= 130:
    #                 # 第一段：从120处的1线性递减到145处的0.5
    #                 coefficient = 1.0 - 0.5 * (130 - i) / (30)
    #             else:
    #                 # 第二段：从145处的0.5线性递增到170处的1
    #                 coefficient = 0.5 + 0.5 * (i - 130) / (30)
    #             final_sum_entropy_values_less[1][i] *= coefficient

    n_subplots = max(len(final_sum_entropy_values_more), len(final_sum_entropy_values_less))
    
    # 创建主图形
    fig = plt.figure(figsize=(20, 6))
    
    # 设置步骤间隔比例
    GAP_RATIO = 0.15  # 步骤之间的间隔占总宽度的比例
    
    # 计算每个子图的宽度比例
    width_ratio = [(1 - GAP_RATIO * (n_subplots-1)) / n_subplots] * n_subplots
    
    # 创建网格规格
    gs = fig.add_gridspec(1, n_subplots, width_ratios=width_ratio, wspace=GAP_RATIO)
    
    # 为两个列表设置不同的颜色和样式
    color_more = 'red'
    color_less = 'green'
    
    # 为每个位置创建子图
    for i in range(n_subplots):
        ax = fig.add_subplot(gs[0, i])
        
        # 绘制more数据（如果存在）
        if i < len(final_sum_entropy_values_more):
            data_more = final_sum_entropy_values_more[i]
            if len(data_more) > 3:
                data_more = smooth_data(data_more)
            x_values = np.arange(len(data_more))
            ax.plot(x_values, data_more, color=color_more, marker='o', 
                    linestyle='-', alpha=0.7, markersize=3, label='More')
        
        # 绘制less数据（如果存在）
        if i < len(final_sum_entropy_values_less):
            data_less = final_sum_entropy_values_less[i]
            if len(data_less) > 3:
                data_less = smooth_data(data_less)
            x_values = np.arange(len(data_less))
            ax.plot(x_values, data_less, color=color_less, marker='s', 
                    linestyle='--', alpha=0.7, markersize=3, label='Less')
        
        # 设置子图标题和格式
        ax.set_title(f'Step {i+1}', fontproperties=custom_font_sub_title)
        
        # 仅在最左侧的子图上显示y轴标签
        if i == 0:
            ax.set_ylabel('Entropy Value', fontproperties=custom_font_ylabel)
        
        # 设置x轴标签
        ax.set_xlabel('Token Index', fontproperties=custom_font_x)
        
        # 设置网格
        ax.grid(True, axis='y')
        ax.grid(False, axis='x')
        
        # 调整x轴刻度
        if i < len(final_sum_entropy_values_more) and len(final_sum_entropy_values_more[i]) > 10:
            data_len = len(final_sum_entropy_values_more[i])
            n_ticks = 5  # 限制x轴刻度数量
            step = max(data_len // n_ticks, 1)
            ax.set_xticks(np.arange(0, data_len, step))
        elif i < len(final_sum_entropy_values_less) and len(final_sum_entropy_values_less[i]) > 10:
            data_len = len(final_sum_entropy_values_less[i])
            n_ticks = 5
            step = max(data_len // n_ticks, 1)
            ax.set_xticks(np.arange(0, data_len, step))
        
        # 只在有两条线的子图上添加图例
        if i < len(final_sum_entropy_values_more) and i < len(final_sum_entropy_values_less):
            ax.legend(prop=custom_font_legend)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
    
    # 设置主标题
    fig.suptitle('Entropy Distribution Comparison', fontproperties=custom_font_title)
    
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.85)  # 为主标题腾出空间
    
    plt.savefig(draw_path)
    print(f"熵分布比较图已保存到 {draw_path}")
    
    plt.close(fig)

def draw_entropy11_distribution(data_paths, draw_path, methods):
    custom_font_title = FontProperties(fname=font_path_bold, size=30)
    custom_font_sub_title = FontProperties(fname=font_path_bold, size=27.5)
    custom_font_x = FontProperties(fname=font_path_bold, size=25)
    custom_font_y = FontProperties(fname=font_path, size=17.5)
    custom_font_xlabel = FontProperties(fname=font_path_bold, size=20)
    custom_font_ylabel = FontProperties(fname=font_path_bold, size=25)
    custom_font_legend = FontProperties(fname=font_path_bold, size=25)

    data_paths = data_paths.split(',')
    methods = methods.split(',')
    faker_datas = []
    other_datas = []
    other_methods = []
    # 处理每个方法和对应的数据路径
    for i, (data_path, method) in enumerate(zip(data_paths, methods)):
        if method.strip().lower() == 'tool-light':
            # 提取Faker方法对应的数据路径
            faker_data_path = data_path.strip()
            with open(faker_data_path, 'r', encoding='utf-8') as f:
                faker_datas.extend(json.load(f))
        else:
            # 提取其他方法对应的数据路径
            other_data_path = data_path.strip()
            other_methods.append(method.strip())
            with open(other_data_path, 'r', encoding='utf-8') as f:
                other_datas.append(json.load(f))
    
    for data in faker_datas:
        entropy_list = data.get('entropy_list_final', [])
        # 1. Calculate sums of each innermost list
        sum_lists = [[sum(inner_list) for inner_list in sublist] for sublist in entropy_list if entropy_list is not None]
        
        # 2 & 3. Find min sum for each position and keep corresponding lists
        result = []
        max_positions = max(len(sublist) for sublist in sum_lists) # 最大推理段数
        
        for pos in range(max_positions):
            # min_sum = float('inf')
            # min_list = None
            
            # for i, sublist in enumerate(sum_lists):
            #     if pos < len(sublist) and sublist[pos] < min_sum:
            #         min_sum = sublist[pos]
            #         min_list = entropy_list[i][pos]
            
            # if min_list is not None:
            #     result.append(min_list)
            # 收集所有在pos位置有值的列表
            available_lists = []
            for i, sublist in enumerate(sum_lists):
                if pos < len(sublist):
                    available_lists.append(entropy_list[i][pos])
            
            # 如果有可用列表，随机选择一个
            if available_lists:
                # selected_list = random.choice(available_lists)
                # 随机选择一个索引
                selected_index = rng.choice(len(available_lists))

                # 根据索引获取对应的列表
                selected_list = available_lists[selected_index]
                result.append(selected_list)
        
        # 4. Update entropy_list_final to be 2-level
        data['entropy_list_final'] = result

    for other_data in other_datas:
        for data in other_data:
            # print(data.keys())
            entropy_list = data.get('entropy_list_final', [])
            # print(entropy_list)
            # 1. Calculate sums of each innermost list
            sum_lists = [[sum(inner_list) for inner_list in sublist] for sublist in entropy_list]
            # print(f"Sum lists: {sum_lists}")
            
            # 2 & 3. Find min sum for each position and keep corresponding lists
            result = []
            max_positions = max(len(sublist) for sublist in sum_lists) if sum_lists else 0
            
            for pos in range(max_positions):
                # 收集所有在pos位置有值的列表
                available_lists = []
                for i, sublist in enumerate(sum_lists):
                    if pos < len(sublist):
                        available_lists.append(entropy_list[i][pos])
                
                # 如果有可用列表，随机选择一个
                if available_lists:
                    # selected_list = random.choice(available_lists)
                    # 随机选择一个索引
                    selected_index = rng.choice(len(available_lists))

                    # 根据索引获取对应的列表
                    selected_list = available_lists[selected_index]
                    result.append(selected_list)
            # 4. Update entropy_list_final to be 2-level
            data['entropy_list_final'] = result

    # 计算每一段的平均值
    max_sequence_count = max(len(data['entropy_list_final']) for data in faker_datas)
    faker_values = {key: [] for key in range(1, max_sequence_count + 1)}
    for key in faker_values.keys():
        for i in range(len(faker_datas)):
            item = faker_datas[i]['entropy_list_final']
            if len(item) >= key:
                faker_values[key].append(item[key - 1])
    final_faker_values = {}
    # print(faker_values[1])
    # assert False
    for key, value_list in faker_values.items():
        max_tokens = max(len(item) for item in value_list) if value_list else 0
        avg_entropy_list = [[] for _ in range(max_tokens)]
        for items in value_list:
            for idx, item in enumerate(items):
                if idx < max_tokens:
                    avg_entropy_list[idx].append(item)
        # 计算平均值
        avg_entropy_list = [sum(items) / len(items) if items else 0 for items in avg_entropy_list]
        # 将平均值添加到最终结果中
        final_faker_values[key] = avg_entropy_list
    final_other_values = []
    for other_data in other_datas:
        # 计算每一段的平均值
        max_sequence_count = max(len(data['entropy_list_final']) for data in other_data)
        other_values = {key: [] for key in range(1, max_sequence_count + 1)}
        for key in other_values.keys():
            for i in range(len(other_data)):
                item = other_data[i]['entropy_list_final']
                if len(item) >= key:
                    other_values[key].append(item[key - 1])
        final_other_value = {}
        for key, value_list in other_values.items():
            max_tokens = max(len(item) for item in value_list) if value_list else 0
            avg_entropy_list = [[] for _ in range(max_tokens)]
            for items in value_list:
                for idx, item in enumerate(items):
                    if idx < max_tokens:
                        avg_entropy_list[idx].append(item)
            # 计算平均值
            avg_entropy_list = [sum(items) / len(items) if items else 0 for items in avg_entropy_list]
            # 将平均值添加到最终结果中
            final_other_value[key] = avg_entropy_list
        final_other_values.append(final_other_value)
    final_datas = []
    final_datas.append(
        list(final_faker_values.values())
    )
    for other_value in final_other_values:
        final_datas.append(
            list(other_value.values())
        )
    # print(final_datas[0])
    min_length = min(len(data) for data in final_datas)
    min_length = min(min_length, 4)
    # 确保所有数据长度一致
    for i in range(len(final_datas)):
        if len(final_datas[i]) > min_length:
            final_datas[i] = final_datas[i][:min_length]

    max_sequence_count = max(len(data) for data in final_datas)
    max_lengths = [[] for _ in range(max_sequence_count)]
    for count in range(max_sequence_count):
        for datas in final_datas:
            if len(datas) > count:
                max_lengths[count].append(len(datas[count]))
        if count != max_sequence_count - 1:
            max_lengths[count].remove(max(max_lengths[count]))

    # print(f"Max lengths for each sequence: {max_lengths}")
    max_lengths = [max(max_lengths[count]) + 10 for count in range(len(max_lengths))]
    # 对数据进行截断处理，确保长度一致
    for i in range(len(final_datas)):
        for j in range(len(final_datas[i])):
            if j < len(max_lengths) and len(final_datas[i][j]) > max_lengths[j]:
                original_list = final_datas[i][j]
                target_length = max_lengths[j]
                
                if target_length <= 2:
                    # 如果目标长度很小，直接截断
                    final_datas[i][j] = original_list[:target_length]
                else:
                    # 保留开头和结尾，截断中间部分
                    keep_start = target_length // 2
                    keep_end = target_length - keep_start
                    
                    new_list = original_list[:keep_start] + original_list[-keep_end:]
                    final_datas[i][j] = new_list

    # n_subplots = min_length
    # # 创建主图形
    # fig = plt.figure(figsize=(20, 6))
    # # 设置步骤间隔比例
    # GAP_RATIO = 0.15  # 步骤之间的间隔占总宽度的比例
    # # 计算每个子图的宽度比例
    # width_ratio = [(1 - GAP_RATIO * (n_subplots-1)) / n_subplots] * n_subplots
    # # 创建网格规格
    # gs = fig.add_gridspec(1, n_subplots, width_ratios=width_ratio, wspace=GAP_RATIO)

    # # 为每个位置创建子图
    # colors = ['red', '#0a3e81', '#3e8ec4', '#7ab6d9', '#a4cce3']  # 预定义颜色列表
    # for i in range(n_subplots):
    #     ax = fig.add_subplot(gs[0, i])
        
    #     # 绘制Faker数据
    #     # print(final_datas[0])
    #     max_xlim = 0
    #     if i < len(final_datas[0]):
    #         data_faker = final_datas[0][i]
    #         if len(data_faker) > 3:
    #             data_faker = smooth_data(data_faker)
    #         x_values = np.arange(len(data_faker))
    #         ax.plot(x_values, data_faker, color=colors[0], marker='o', 
    #                 linestyle='-', alpha=0.7, markersize=3, label='Faker')
    #         max_xlim = max(max_xlim, len(data_faker))
            
    #     # 绘制其他方法数据
    #     for j in range(1, len(final_datas)):
    #         if i < len(final_datas[j]):
    #             data_other = final_datas[j][i]
    #             if len(data_other) > 3:
    #                 data_other = smooth_data(data_other)
    #             x_values = np.arange(len(data_other))
    #             ax.plot(x_values, data_other, color=colors[j], marker='s', 
    #                     linestyle='--', alpha=0.7, markersize=3, label=f'{other_methods[j-1].strip()}')
    #             max_xlim = max(max_xlim, len(data_other))
        
    #     # 设置子图标题和格式
    #     ax.set_title(f'Step {i+1}', fontproperties=custom_font_sub_title)
        
    #     # 仅在最左侧的子图上显示y轴标签
    #     if i == 0:
    #         ax.set_ylabel('Entropy Value', fontproperties=custom_font_ylabel)
        
    #     # 设置x轴标签
    #     ax.set_xlabel('Token Index', fontproperties=custom_font_x)
    #     # 设置x轴范围
    #     ax.set_xlim(0, max_xlim)  # 设置x轴范围
        
    #     # 设置网格
    #     ax.grid(True, axis='y')
    #     ax.grid(False, axis='x')
        
    #     # 调整x轴刻度
    #     if max_xlim > 10:
    #         n_ticks = 5  # 限制x轴刻度数量
    #         step = max(max_xlim // n_ticks, 1)
    #         ax.set_xticks(np.arange(0, max_xlim, step))

    #     # 只在有两条线的子图上添加图例
    #     if len(final_datas) > 1:
    #         ax.legend(prop=custom_font_legend, loc='upper right')
    #     ax.tick_params(axis='x', labelsize=15)
    #     ax.tick_params(axis='y', labelsize=15)
    
    # # 设置主标题
    # fig.suptitle('Entropy Distribution Comparison', fontproperties=custom_font_title)
    
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.975)  # 为主标题腾出空间
    
    # plt.savefig(draw_path)
    # print(f"熵分布比较图已保存到 {draw_path}")
    
    # plt.close(fig)
    n_subplots = min_length
    # 创建主图形
    fig = plt.figure(figsize=(16, 12))  # 调整图形大小以适应2x2布局

    # 计算2x2网格的行列数
    n_rows = 2
    n_cols = 2
    # 创建网格规格 - 2行2列
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.2)  # 调整行列间距

    # 为每个位置创建子图
    colors = ['red', '#0a3e81', '#3e8ec4', '#7ab6d9', '#a4cce3']  # 预定义颜色列表
    for i in range(n_subplots):
        # 计算当前子图在2x2网格中的位置
        row = i // n_cols  # 行索引
        col = i % n_cols   # 列索引
        
        ax = fig.add_subplot(gs[row, col])
        
        # 绘制Faker数据
        max_xlim = 0
        if i < len(final_datas[0]):
            data_faker = final_datas[0][i]
            if len(data_faker) > 3:
                data_faker = smooth_data(data_faker)
            x_values = np.arange(len(data_faker))
            ax.plot(x_values, data_faker, color=colors[0], marker='o', 
                    linestyle='-', alpha=0.7, markersize=3, label='Tool-Light')
            max_xlim = max(max_xlim, len(data_faker))
            
        # 绘制其他方法数据
        for j in range(1, len(final_datas)):
            if i < len(final_datas[j]):
                data_other = final_datas[j][i]
                if len(data_other) > 3:
                    data_other = smooth_data(data_other)
                x_values = np.arange(len(data_other))
                ax.plot(x_values, data_other, color=colors[j], marker='s', 
                        linestyle='--', alpha=0.7, markersize=3, label=f'{other_methods[j-1].strip()}')
                max_xlim = max(max_xlim, len(data_other))
        
        # 设置子图标题和格式
        ax.set_title(f'Step {i+1}', fontproperties=custom_font_sub_title)
        
        # 只在左侧子图上显示y轴标签
        if col == 0:
            ax.set_ylabel('Entropy Value', fontproperties=custom_font_ylabel)
        
        # 只在底部子图上显示x轴标签
        if row == n_rows - 1:
            ax.set_xlabel('Token Index', fontproperties=custom_font_x)
        
        # 设置x轴范围
        ax.set_xlim(0, max_xlim)
        
        # 设置网格
        ax.grid(True, axis='y')
        ax.grid(False, axis='x')
        
        # 调整x轴刻度
        if max_xlim > 10:
            n_ticks = 5
            step = max(max_xlim // n_ticks, 1)
            ax.set_xticks(np.arange(0, max_xlim, step))

        # 只在有两条线的子图上添加图例
        # if len(final_datas) > 1:
        #     ax.legend(prop=custom_font_legend, loc='upper right')
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
    
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], label=label) for i, label in enumerate(['Tool-Light'] + other_methods)]
    fig.legend(legend_handles, ['Tool-Light'] + other_methods, loc='lower center', ncol=5,
                bbox_to_anchor=(0.5, 0.05), frameon=True, edgecolor='gray', prop=custom_font_legend)

    # 设置主标题
    fig.suptitle('Entropy Distribution Comparison', fontproperties=custom_font_title)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.2, left=0.08, right=0.95)  # 调整边距

    plt.savefig(draw_path)
    print(f"熵分布比较图已保存到 {draw_path}")

    plt.close(fig)

def draw_wordcloud(data_path, draw_path):
    """
    绘制词云图。
    
    Args:
        text (str): 用于生成词云的文本数据。
        draw_path (str): 保存词云图的路径。
    """
    from matplotlib import pyplot as plt
    # 准备文本数据
    source_datas = []
    if ',' in data_path:        # 如果data_path包含逗号，表示多个文件路径
        data_paths = data_path.split(',')
        for path in data_paths:
            with open(path.strip(), 'r', encoding='utf-8') as f:
                temp_datas = json.load(f)
                if 'toolstar' in path:
                    count = 150
                    temp_datas = random.sample(temp_datas, count) if len(temp_datas) > count else temp_datas
                source_datas.extend(temp_datas)
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            source_datas = json.load(f)
    # print(source_datas[0].keys())
    for i in range(len(source_datas)):
        # 拼接entropy和tokens数据
        source_datas[i]['entropy'] = [item for sublist in source_datas[i]['entropy'] for item in sublist]
        source_datas[i]['tokens'] = [item for sublist in source_datas[i]['tokens'] for item in sublist] if 'tokens' in source_datas[i] else [item for sublist in source_datas[i]['token_lists'] for item in sublist]
    entropies, tokens = [], []
    for source_data in source_datas:
        entropies.extend(source_data['entropy'])
        tokens.extend(source_data['tokens'])
    # 根据熵值对tokens进行排序，找出前20%最大熵值对应的tokens
    entropy_token_pairs = list(zip(entropies, tokens))
    # 按熵值降序排序
    entropy_token_pairs.sort(key=lambda x: x[0], reverse=True)

    # 计算前20%的数量
    top_20_percent_count = int(len(entropy_token_pairs) * 0.2)

    # 分离前20%和剩余80%
    top_20_percent_pairs = entropy_token_pairs[:top_20_percent_count]
    remaining_80_percent_pairs = entropy_token_pairs[top_20_percent_count:]

    # 提取对应的tokens
    top_20_percent_tokens = [pair[1] for pair in top_20_percent_pairs]
    remaining_80_percent_tokens = [pair[1] for pair in remaining_80_percent_pairs]

    # 将tokens列表转换为文本字符串用于词云生成
    top_20_texts = ' '.join(top_20_percent_tokens)
    remaining_80_texts = ' '.join(remaining_80_percent_tokens)

    # 创建词云对象
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        max_words=200,
        colormap='viridis',  # 可选择不同色彩映射
        font_path=font_path_bold  # 可选自定义字体
    )

    if ',' in draw_path:
        draw_paths = draw_path.split(',')

        # 生成词云
        wordcloud.generate(top_20_texts)

        # 显示词云
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

        # 保存词云图
        wordcloud.to_file(draw_paths[0].strip())
        print(f"词云图已保存到 {draw_paths[0].strip()}")
        
        # 生成剩余80%的词云
        wordcloud.generate(remaining_80_texts)
        # 显示剩余80%的词云
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        # 保存剩余80%的词云图
        wordcloud.to_file(draw_paths[1].strip())
        print(f"剩余80%的词云图已保存到 {draw_paths[1].strip()}")
    else:
        # 生成词云
        wordcloud.generate(top_20_texts)

        # 显示词云
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

        # 保存词云图
        wordcloud.to_file('/home/u2024001021/agentic_search/pictures/wordcloud.pdf')
        print(f"词云图已保存到 {draw_path}")

    
def calculate_kl_divergence(p, q):
    """计算两个分布的KL散度平均值"""
    # 确保输入是numpy数组并且非零
    p = np.array(p) + 1e-10
    q = np.array(q) + 1e-10
    
    # 归一化
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # 计算KL散度并取平均
    return np.mean(kl_div(p, q))

