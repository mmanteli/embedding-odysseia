#!/bin/bash

#SBATCH -A project_462000883
#SBATCH -p dev-g
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -J hidden-states
#SBATCH -o logs/%x-%j.out

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.4
export PYTHONPATH=/scratch/project_462000883/amanda/embedding-odysseia/pythonuserbase/lib/python3.10/site-packages:$PYTHONPATH
export HF_HOME=/scratch/project_462000883/hf_cache

register="dtp"
iter="50k"
data="/scratch/project_462000883/amanda/register-training-with-megatron/data_by_checkpoints/dtp-50k-51k.jsonl"
data_name="dtp-50k-51k"
srun python extract.py --register=$register --iter=$iter --data_path=$data --data_is_tokenized --sample=10 --save="results/${register}-${iter}-w-${data_name}.tsv"