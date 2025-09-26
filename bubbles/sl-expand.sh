#!/bin/bash

#SBATCH -A project_462000883
#SBATCH -p dev-g
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -J expand
#SBATCH -o logs/%x-%j.out

index=$1

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.7
export PYTHONPATH=/scratch/project_462000883/amanda/pythonuserbase/lib/python3.10/site-packages:$PYTHONPATH
export HF_HOME=/scratch/project_462000883/hf_cache

echo $(date +"%T")
python expand.py --embedding_file="seed_embeddings.jsonl" --file_index=$index --savepath="tests/" --delta="0.05" --batch_size=32
echo $(date +"%T")
