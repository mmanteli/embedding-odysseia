#!/bin/bash

#SBATCH -A project_2002026
#SBATCH -p gpusmall
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -t 02:30:00
#SBATCH -N 1
#SBATCH -J extract
#SBATCH -o logs/%x-%j.out

register=$1
model="e5"
split_by="truncate"
data_to_embed="/scratch/project_2009498/register-data/eng_Latn_${register}.jsonl"

module load pytorch
export HF_HOME=/scratch/project_2002026/amanda/hf_cache
echo Starting at $(date +%H:%M.%S)
srun python extract.py --model=$model \
                       --split_by=$split_by \
                       --save="/scratch/project_2002026/amanda/from-lumi/embedded/$model/${split_by}/${register}.pkl" \ 
                       < $data_to_embed
echo Ending at $(date +%H:%M.%S)