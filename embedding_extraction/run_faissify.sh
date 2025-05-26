#!/bin/bash

#SBATCH -A project_2002026
#SBATCH -p small
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -t 06:30:00
#SBATCH -N 1
#SBATCH -J faissify
#SBATCH -o logs/%x-%j.out

module load pytorch
export HF_HOME=/scratch/project_2002026/amanda/hf_cache
path="/scratch/project_2002026/amanda/from-lumi/embedding-odysseia"
run_name="test"
srun python faissify.py --base_indexer="IVFPQ" \
                        --trained_indexer="${path}/${run_name}_trained.index" \
                        --filled_indexer="${path}/${run_name}_filled.index" \
                        --data="/scratch/project_2002026/amanda/from-lumi/embedded/e5/truncate/OP.jsonl"
                        --database="${path}/${run_name}.sqlite"