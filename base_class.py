import ujson
from vllm import SamplingParams
from tqdm import tqdm
from prompts import *
from utils import *

class BaseClass():
    def __init__(self, data_path=None, output_path=None, model=None, tokenizer=None, params_config=None, counts=200):
        self.data_path = data_path
        self.output_path = output_path
        self.model = model
        self.tokenizer = tokenizer
        self.params_config = params_config
        self.counts = counts
        self.prompt = """
You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool and python interpreter tool. \
Given a question, you need to first think about the reasoning process in the mind and then provide the answer. \
During thinking, you can invoke the wikipedia search tool to search and python interpreter tool to calculate the math problem for fact information about specific topics if needed. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, \
and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. \
For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> \
<think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> \
<think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. \
In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format.
"""
        if params_config and isinstance(self.params_config, dict):
            self.params_config = SamplingParams(**self.params_config)
        self.source_datas = []
        self.load_data()

    def load_data(self, path=None):
        if self.data_path is None and path is None:
            return
        if path is not None:
            self.data_path = path
        print(f"Loading data from {self.data_path} ...")
        try:
            if ',' in self.data_path:
                self.data_paths = self.data_path.split(',')
                self.source_datas = []
                for path in self.data_paths:
                    if path.endswith('.json'):
                        with open(path, 'r', encoding='utf-8') as f:
                            self.source_datas.extend(ujson.load(f))
                        print(f"Loaded {len(self.source_datas)} records from {path}.")
                    elif path.endswith('.jsonl'):
                        with open(path, 'r', encoding='utf-8') as f:
                            for line in f:
                                self.source_datas.append(ujson.loads(line))
                        print(f"Loaded {len(self.source_datas)} records from {path}.")
                # random.shuffle(self.source_datas)
                if self.counts > 0:
                    self.source_datas = self.source_datas[:self.counts]
            elif self.data_path.endswith('.json'):
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    self.source_datas = ujson.load(f)
                print(f"Loaded {len(self.source_datas)} records from {self.data_path}.")
                if self.counts > 0:
                    self.source_datas = self.source_datas[:self.counts]  # 限制数据量
            elif self.data_path.endswith('.jsonl'):
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        self.source_datas.append(ujson.loads(line))
                print(f"Loaded {len(self.source_datas)} records from {self.data_path}.")
                if self.counts > 0:
                    self.source_datas = self.source_datas[:self.counts]  # 限制数据量
        except:
            print(f"Failed to load data from {self.data_path}. Please check the file format or path.")

    def run(self):
        raise NotImplementedError("The run method must be implemented in the subclass.")

    def filter_long_datas(self, data_list, max_tokens=6000):
        filtered_data = []
        for i in tqdm(range(len(data_list))):
            item = data_list[i]
            chosen = item['chosen'] if 'chosen' in item else item['output']
            if len(self.tokenizer.encode(QWEN_TEMPLATE.format(prompt=SFT_PROMPT, question=item['input']) + chosen)) <= max_tokens and validate_format(chosen):
                filtered_data.append(item)
        return filtered_data
