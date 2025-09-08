#!/bin/bash

#SBATCH -A project_462000883
#SBATCH -p dev-g
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -J GEIA
#SBATCH -o logs/%x-%j.out

project="project_462000883"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
export PYTHONPATH=/scratch/project_462000883/amanda/pythonuserbase/lib/python3.10/site-packages:$PYTHONPATH
export HF_HOME=/scratch/project_462000883/hf_cache


#GIEA
#GPT-2 Attacker
#You need to set up arguments properly before running codes: python attacker.py

#--model_dir: Attacker model path from Huggingface (like 'gpt2-large' and 'microsoft/DialoGPT-xxxx') or local model checkpoints.
#--num_epochs: Training epoches.
#--batch_size: Batch_size #.
#--dataset: Name of the dataset including personachat, qnli, mnli, sst2, wmt16, multi_woz and abcd.
#--data_type: train or dev or test.
#--embed_model: The victim model you wish to attack. We currently support sentence-bert models and huggingface models, you may refer to our model_cards dictionary in attacker.py for more information.
#--decode: Decoding algorithm. We currently implement beam and sampling based decoding.
#You should train the attacker on training data at first, then test your attacker on the test data to obtain test logs. Then you can evaluate attack performance on test logs by changing model_dir to your trained attcker and data_type to test.

#If you want to train a randomly initialized GPT-2 attacker, after setting the arguments, run: python attacker_random_gpt2.py

model="gpt-2"
dataset="mnli"
split="train"
embedding_model="all-roberta-large-v1"

python GEIA/attacker.py \
    --model_dir ${model} \
    --num_epochs: 10 \
    --batch_size: 32 \
    --dataset: ${dataset} \
    --data_type: ${split} \
    --embed_model: ${embedding_model} \
    --decode beam