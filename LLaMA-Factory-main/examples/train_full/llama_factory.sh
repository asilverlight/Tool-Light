module load cuda/12.1.1
cd /path/to/LLaMA-Factory-main
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train /path/to/LLaMA-Factory-main/examples/train_full/sft.yaml

CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train /path/to/LLaMA-Factory-main/examples/train_full/step1_dpo.yaml

CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train /path/to/LLaMA-Factory-main/examples/train_full/step2_dpo.yaml