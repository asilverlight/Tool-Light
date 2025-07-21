from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os
import argparse
 
def merge_lora(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, padding_side="left")
 
    print(f"Loading the LoRA adapter from {lora_path}")
 
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
    )
 
    print("Applying the LoRA")
    model = lora_model.merge_and_unload()
 
    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA model with base model")
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model")
    parser.add_argument("--lora_model", type=str, required=True, help="Path to the LoRA model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged model")
    args = parser.parse_args()
    merge_lora(
        model_name_or_path=args.base_model,
        lora_path=args.lora_model,
        output_path=args.output_path,
    )