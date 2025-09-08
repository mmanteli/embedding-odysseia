#!/bin/bash

#SBATCH -A project_462000883
#SBATCH -p small-g
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -t 01:44:00
#SBATCH -N 1
#SBATCH -J v2t
#SBATCH -o logs/%x-%j.out


num_sentences=$1
search_depth=$2

register=$3

case $register in
    fake_texts|fake_texts_long|fake_texts_punct)
        file="/scratch/project_462000883/amanda/inversion_repos/fake_sentence_dataset/${register}.jsonl"
        num_sentences=0 # force this
    ;;
    *)
        file="/scratch/project_462000883/amanda/register-data/${register}.jsonl"
    ;;
esac
echo $file


module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
export PYTHONPATH=/scratch/project_462000883/amanda/pythonuserbase/lib/python3.10/site-packages:$PYTHONPATH
export HF_HOME=/scratch/project_462000883/hf_cache

echo "test_vec2text.py --files=$file --length=${num_sentences} --search_iter=${search_depth}"
python test_vec2text.py --files=$file --length=${num_sentences} --search_iter=${search_depth}

mv logs/$SLURM_JOB_NAME-$SLURM_JOB_ID.out "logs/v2t/${register}_sentences_${num_sentences}_iters_${search_depth}.out"
