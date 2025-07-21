from vllm import SamplingParams
from tqdm import tqdm
import torch
from ..prompts import *
from .utils import *
import ujson
from ..args import parse_args
from ..base_class import BaseClass # type: ignore

class DirectInferenceToolStarSFT(BaseClass):
    def __init__(self, data_path=None, output_path=None, model=None, tokenizer=None, params_config=None, counts=-1):
        super().__init__(data_path, output_path, model, tokenizer, params_config, counts)

    def run(self):
        """
        Run the inference process.
        """
        inputs = []
        questions, answers = [], []
        tool_star_outputs = []
        for i in range(len(self.source_datas)):
            instruction = WITHOUT_TOOL
            question = self.source_datas[i]['input']
            tool_star_outputs.append(self.source_datas[i]['output'])
            answer = extract_answer(self.source_datas[i]['output'])
            questions.append(question)
            answers.append(answer)
            inputs.append(
                self.tokenizer.apply_chat_template(
                    [
                        {
                            'role': 'system', 'content': instruction
                        },
                        {
                            'role': 'user', 'content': question
                        }
                    ], tokenize=False, add_generation_prompt=True
                )
            )
        outputs = self.model.generate(
            inputs,
            self.params_config,
        )
        results = []
        for i in range(len(outputs)):
            texts = [sequence.text.strip() for sequence in outputs[i].outputs]
            results.append(
                {
                    'question': questions[i],
                    'tool_star_output': tool_star_outputs[i],
                    'golden_answer': answers[i],
                    'outputs': texts,
                }
            )
        if self.output_path is not None:
            with open(self.output_path, 'w') as f:
                ujson.dump(results, f, indent=4, ensure_ascii=False)
            print(f"Results saved to {self.output_path}")

if __name__ == '__main__':
    args = parse_args()
    model_path = model2path[args.model_name]
    model_config = {
        'model_path': model_path,
        'type': torch.bfloat16,
        'max_input_len': args.max_input_len,
        'gpu_use': args.gpu_use,
        'gpu_num': torch.cuda.device_count(),
    }
    model, tokenizer = load_model(model_config)
    params_config = {
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'top_p': 0.95,
        'top_k': 40,
        'repetition_penalty': 1.1,
        'n': 5,
    }
    directinferencetoolstarsft = DirectInferenceToolStarSFT(
        data_path=args.data_path,
        output_path=args.output_path,
        model=model,
        tokenizer=tokenizer,
        params_config=params_config,
        counts=args.counts
    )
    directinferencetoolstarsft.run()

