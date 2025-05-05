#!/bin/bash

#SBATCH -A project_462000883
#SBATCH -p small-g
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -t 02:30:00
#SBATCH -N 1
#SBATCH -J extract
#SBATCH -o logs/%x-%j.out

register=$1
module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.4
export PYTHONPATH=/scratch/project_462000883/amanda/embedding-extraction/pythonuserbase/lib/python3.10/site-packages:$PYTHONPATH

export HF_HOME=/scratch/project_462000883/hf_cache
echo Starting at $(date +%H:%M.%S)
srun python extract.py --model="e5" --split_by="sentences" --save="/scratch/project_462000883/amanda/embedded-data/e5/${register}_test.pkl" < /scratch/project_462000353/HPLT-REGISTERS/samples-150B-by-register-xlmrl/original_corrected/eng_Latn_${register}.jsonl
echo Ending at $(date +%H:%M.%S)