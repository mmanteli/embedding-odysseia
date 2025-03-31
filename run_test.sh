#!/bin/bash

#SBATCH -A project_462000615
#SBATCH -p dev-g
#SBATCH --ntasks-per-node=1
##SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH -J test_sentence_transformers
#SBATCH -o logs/%x-%j.out

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.4
export PYTHONPATH=/scratch/project_462000615/mynttiam/embedding-extraction/pythonuserbase/lib/python3.10/site-packages:$PYTHONPATH

export HF_HOME=/scratch/project_462000615/hf_cache
srun python extract_embeddings.py --data="/scratch/project_462000353/HPLT-REGISTERS/samples-150B-by-register-xlmrl/original_corrected/IN-splitted/eng_Latn_IN_121_01.jsonl" --model="e5" --save_path="testi.jsonl"