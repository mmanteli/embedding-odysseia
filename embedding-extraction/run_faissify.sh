#!/bin/bash

#SBATCH -A project_462000883
#SBATCH -p small
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -t 06:30:00
#SBATCH -N 1
#SBATCH -J faissify
#SBATCH -o logs/%x-%j.out

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.4
export PYTHONPATH=/scratch/project_462000883/amanda/embedding-extraction/pythonuserbase/lib/python3.10/site-packages:$PYTHONPATH
export HF_HOME=/scratch/project_462000883/hf_cache

run_name="dtp-ne-OP-SP"
srun python faissify.py --base_indexer="IVFPQ" \
                        --trained_indexer="/scratch/project_462000883/amanda/embedding-extraction/trained-indexers/${run_name}.index" \
                        --filled_indexer="/scratch/project_462000883/amanda/embedding-extraction/filled-indexers/${run_name}.index" \
                        --data='/scratch/project_462000883/amanda/embedded-data/e5/dtp_test.pkl,/scratch/project_462000883/amanda/embedded-data/e5/ne_test.pkl,/scratch/project_462000883/amanda/embedded-data/e5/OP_test.pkl,/scratch/project_462000883/amanda/embedded-data/e5/SP_test.pkl' \
                        --training_data="/scratch/project_462000883/amanda/embedding-extraction/training-data/${run_name}.pt" \
                        --database="indexed-data/${run_name}.sqlite"