import ujson
from tqdm import tqdm
from args import parse_args
from base_class import BaseClass
from prompts import *
from utils import *

import math

class CalculateMetrics(BaseClass):
    def __init__(self, data_path=None, output_path=None, model=None, tokenizer=None, params_config=None, counts=-1, other_paths=None, dataset=None):
        super().__init__(data_path, output_path, model, tokenizer, params_config, counts)
        self.other_paths = other_paths
        self.dataset = dataset

    def load_other_data(self, other_path):
        with open(other_path, 'r') as f:
            results = ujson.load(f)
        return results

    def efficiency(self):
        if not self.other_paths:
            raise ValueError("Other paths are not set.")
        
        other_datas = []
        self.other_paths = self.other_paths.split(',')
        for path in self.other_paths:
            other_datas.append(self.load_other_data(path))
        results = {}

        for i in tqdm(range(len(other_datas))):
            path = self.other_paths[i]
            start_idx = path.find('path/to/')
            if start_idx != -1:
                start_idx += len('path/to/')
                next_slash_idx = path.find('/', start_idx)
                if next_slash_idx != -1:
                    key = path[start_idx:next_slash_idx]
                else:
                    key = path[start_idx:]
            else:
                key = self.other_paths[i]
            results[key] = []
            epison = 0.0
            for j in range(len(other_datas[i])):
                data = other_datas[i][j]
                if self.dataset in ['hotpotqa', '2wiki', 'musique','bamboogle']:
                    if 'dotamath' not in path.lower() and data['Metrics']['tool_counts'] > 0:
                        results[key].append(
                            data['Metrics']['llm_equal'] / (data['Metrics']['tool_counts'] + epison)
                        )
                    else:
                        results[key].append(
                            data['Metrics']['llm_equal'] / (1 + epison)
                        )
                elif data['Metrics']['tool_counts'] > 0:
                    results[key].append(
                        data['Metrics']['llm_equal'] / (data['Metrics']['tool_counts'] + epison)
                    )
            results[key] = round(sum(results[key]) / len(results[key]), 4)
                
        if self.output_path:
            with open(self.output_path, 'w') as f:
                ujson.dump(results, f, indent=4)
            print(f"Results saved to {self.output_path}")

    def necessity(self):
        if not self.other_paths:
            raise ValueError("Other paths are not set.")
        
        other_datas = []
        self.other_paths = self.other_paths.split(',')
        for path in self.other_paths:
            other_datas.append(self.load_other_data(path))
        datacounts = len(other_datas[0])
        processed_datas = [[] for _ in range(datacounts)]
        scores = [[0 for _ in range(len(other_datas))] for _ in range(datacounts)]
        for i in range(len(other_datas)):
            for j in range(datacounts):
                processed_datas[j].append(other_datas[i][j])
        for i in range(len(processed_datas)):
            for j in range(len(processed_datas[i])):
                data = processed_datas[i][j]
                if data['Metrics']['llm_response'] == 'Incorrect':
                    continue
                for k in range(len(processed_datas[i])):
                    if i == k:
                        continue
                    other_data = processed_datas[i][k]
                    if other_data['Metrics']['llm_response'] == 'Incorrect' and data['Metrics']['tool_counts'] > other_data['Metrics']['tool_counts']:
                        scores[i][j] += 1
                    elif other_data['Metrics']['llm_response'] == 'Correct' and data['Metrics']['tool_counts'] > other_data['Metrics']['tool_counts']:
                        scores[i][j] -= 1
        results = {}
        scores = list(map(list, zip(*scores)))
        for i in range(len(scores)):
            path = self.other_paths[i]
            start_idx = path.find('path/to/')
            if start_idx != -1:
                start_idx += len('path/to/')
                next_slash_idx = path.find('/', start_idx)
                if next_slash_idx != -1:
                    key = path[start_idx:next_slash_idx]
                else:
                    key = path[start_idx:]
            else:
                key = self.other_paths[i]
            results[key] = round(sum(scores[i]) / len(scores[i]), 4)

        # Apply min-max normalization to make all values greater than zero
        if results:
            min_val = min(results.values())
            max_val = max(results.values())
            if max_val != min_val:
                for key in results:
                    results[key] = round((results[key] - min_val) / (max_val - min_val), 4)
            else:
                for key in results:
                    results[key] = 0.0001
                
        if self.output_path:
            with open(self.output_path, 'w') as f:
                ujson.dump(results, f, indent=4)
            print(f"Results saved to {self.output_path}")

if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    other_paths = args.other_paths
    output_path = args.output_path
    exp_type = args.exp_type
    model_path = args.model_path
    dataset = args.dataset
    if exp_type == 'efficiency':
        calculate_metrics = CalculateMetrics(data_path=data_path, output_path=output_path, other_paths=other_paths, dataset=dataset, model=model_path)
        calculate_metrics.efficiency()
    elif exp_type == 'necessity':
        calculate_metrics = CalculateMetrics(data_path=data_path, output_path=output_path, other_paths=other_paths, dataset=dataset, model=model_path)
        calculate_metrics.necessity()
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}. Supported types: efficiency, necessity.")