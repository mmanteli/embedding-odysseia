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
export PYTHONPATH=/scratch/project_462000883/amanda/embedding-odysseia/pythonuserbase/lib/python3.10/site-packages:$PYTHONPATH
export HF_HOME=/scratch/project_462000883/hf_cache
metric=$1

srun python embedding_extraction/find_path.py --straight \
                        --start="Creme de Methe cake. Simple recipe for my favourite cake." \
                        --target="Visualisation of data in python made easy. Having trouble with data visualisation? In this tutorial we will learn about how to print Data in Tabular Format in Python." \
                        --model="e5" \
                        --metric=$metric \
                        --filled_indexer="/scratch/project_462000883/amanda/embedding-odysseia/jobs/full-docs-29-04-2025/filled-indexers/full_docs-IVFPQ.index" \
                        --database="/scratch/project_462000883/amanda/embedding-odysseia/jobs/full-docs-29-04-2025/filled-indexers/full_docs-IVFPQ.sqlite"


# In this tutorial we will learn about how to print Data in Tabular Format in Python