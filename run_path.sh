#!/bin/bash

#SBATCH -A project_462000883
#SBATCH -p debug
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -J pathing
#SBATCH -o logs/%x-%j.out


module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.4
export PYTHONPATH=/scratch/project_462000883/amanda/embedding-extraction/pythonuserbase/lib/python3.10/site-packages:$PYTHONPATH
export HF_HOME=/scratch/project_462000883/hf_cache

srun python find_path.py --model="e5" \
                         --filled_indexer="/scratch/project_462000883/amanda/embedding-extraction/filled-indexers/dtp-OP.index" \
                         --database="/scratch/project_462000883/amanda/embedding-extraction/indexed-data/dtp-OP.sqlite"
