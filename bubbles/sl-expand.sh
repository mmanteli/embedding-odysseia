#!/bin/bash

#SBATCH -A project_462000883
#SBATCH -p small-g
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -t 12:29:00
#SBATCH -N 1
#SBATCH -J expand_synthetic2
#SBATCH -o logs/%x-%j.out

index=$1

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.7
export PYTHONPATH=/scratch/project_462000883/amanda/pythonuserbase/lib/python3.10/site-packages:$PYTHONPATH
export HF_HOME=/scratch/project_462000883/hf_cache

echo $(date +"%T")
python expand.py --embedding_file="seed_embeddings_synthetic.jsonl" --file_index=$index --savepath="bubbledata/synthetic2/" --delta="0.0005" --batch_size=64
echo $(date +"%T")
