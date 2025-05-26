#!/bin/bash

#SBATCH -A project_2002026
#SBATCH -p debug
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -J sanity
#SBATCH -o logs/%x-%j.out


module load pytorch
export HF_HOME=/scratch/project_2002026/amanda/hf_cache

run_name="test"
srun python sanity_check.py --model="e5" \
                     --filled_indexer="${path}/${run_name}_filled.index" \
                     --database="${path}/${run_name}.sqlite"