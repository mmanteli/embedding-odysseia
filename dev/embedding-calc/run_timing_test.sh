#!/bin/bash

#SBATCH -A project_462000883
#SBATCH -p small-g
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -J time_modelbs_128
#SBATCH -o logs/%x-%j.out

model_bs=128
module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.4
export PYTHONPATH=/scratch/project_462000883/amanda/embedding-extraction/pythonuserbase/lib/python3.10/site-packages:$PYTHONPATH

export HF_HOME=/scratch/project_462000883/hf_cache
#echo Starting 64 at $(date +%H:%M.%S)

for data_batch in 64 128 200 320 520 720 920 1200; do
    start=$(date +%s)
    srun python extract.py --model="e5" --batchsize=$data_batch --split_by="sentences" --model_batch_size=$model_bs --save="testi.pkl" < /scratch/project_462000883/amanda/data/tests/NA_1000.jsonl
    end=$(date +%s)
    difference="$(($end - $start))"
    echo Model inside batch $model_bs, data batch $data_batch: $difference
done

