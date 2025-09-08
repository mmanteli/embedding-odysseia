#!/bin/bash

#SBATCH -A project_462000883
#SBATCH -p small-g
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -t 01:29:00
#SBATCH -N 1
#SBATCH -J adversial_decode_run2
#SBATCH -o logs/%x-%j.out



module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
export PYTHONPATH=/scratch/project_462000883/amanda/pythonuserbase/lib/python3.10/site-packages:$PYTHONPATH
export HF_HOME=/scratch/project_462000883/hf_cache

cd adversarial_decoding
#cd adversial_decoding
python main.py \
    --experiment emb_inv \
    --encoder_name gte 


#    --beam_width $1 \
#    --max_steps $2 \
#    --top_k 10 \
#    --top_p 1
