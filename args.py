import argparse

def parse_args():
    argument_parser = argparse.ArgumentParser(description="Agentic Search Arguments")
    
    argument_parser.add_argument("--data_path", type=str, default=None, help="Path to the input data file.")
    argument_parser.add_argument("--output_path", type=str, default=None, help="Path to save the output data file.")
    argument_parser.add_argument("--model_name", type=str, default="qwen2-7b-instruct", help="Name of the model to use.")
    argument_parser.add_argument("--gpu_use", type=float, default=0.95, help="Fraction of GPU memory to use.")
    argument_parser.add_argument("--max_tokens", type=int, default=32768, help="Maximum number of tokens to generate.")
    argument_parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling.")
    argument_parser.add_argument("--max_input_len", type=int, default=8192, help="Maximum number of input tokens.")
    argument_parser.add_argument("--log_path", type=str, default=None, help="Path to save the log file.")
    argument_parser.add_argument("--counts", type=int, default=-1, help="Number of test datas.")
    argument_parser.add_argument("--exp_type", type=str, default="default", help="Type of experiment to run.")
    argument_parser.add_argument("--sft_path", type=str, default=None, help="Path to the SFT model.")
    argument_parser.add_argument("--dpo_path", type=str, default=None, help="Path to the DPO model.")
    argument_parser.add_argument("--entropy_path", type=str, default=None, help="Path to the entropy model.")
    argument_parser.add_argument("--special_token_keys", type=str, default=None, help="Keys for special tokens.")
    argument_parser.add_argument("--initial_path", type=str, default=None, help="Path to the initial model.")
    argument_parser.add_argument("--rl_path", type=str, default=None, help="Path to the RL model.")
    argument_parser.add_argument("--model_path", type=str, default=None, help="Path to the model.")
    argument_parser.add_argument("--math_method", type=str, default="default", help="Method for handling math operations.")
    argument_parser.add_argument("--dataset", type=str, default="default", help="Dataset to use for training or evaluation.")
    argument_parser.add_argument("--picture_path", type=str, default=None, help="Path to the picture data.")
    argument_parser.add_argument("--other_paths", type=str, default=None, help="Other paths for additional data.")

    args = argument_parser.parse_args()
    return args