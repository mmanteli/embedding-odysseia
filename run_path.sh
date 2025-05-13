#!/bin/bash

#SBATCH -A project_462000883
#SBATCH -p debug
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -J pathing
#SBATCH -o logs/%x-%j.out


module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.4
export PYTHONPATH=/scratch/project_462000883/amanda/embedding-odysseia/pythonuserbase/lib/python3.10/site-packages:$PYTHONPATH
export HF_HOME=/scratch/project_462000883/hf_cache
metric=$1
jobname="full-docs-07-05-25"

full_docs_start=""
full_docs_target=""
srun python embedding_extraction/find_path.py --straight \
                        --start=$full_docs_start \
                        --target=$full_docs_target \
                        --model="e5" \
                        --metric=$metric \
                        --save_plots=${jobname}_3 \
                        --filled_indexer="/scratch/project_462000883/amanda/embedding-odysseia/jobs/${jobname}/filled-indexers/IVFPQ.index" \
                        --database="/scratch/project_462000883/amanda/embedding-odysseia/jobs/${jobname}/filled-indexers/IVFPQ.sqlite"


# In this tutorial we will learn about how to print Data in Tabular Format in Python



# The Regency Place Name Generator -- this generator produces quality place names geared for Regency-era stories, but could work in other time periods as well. # IN dtp
# 'However, it is essential to speak to a doctor who can ease a person’s worries, determine why they have memory loss, and offer a range of treatment options. # IN dtp
# I hope you all like it,\nits my usual mix of Crunchy Metal Rock-n-Roll and Bloody Carnage! # ID
# Understanding this will help you look for a platform that can meet those needs. # OP av
# Also, not everyone on the client side may be well versed with the technicalities of SEO. # OP av