#!/bin/bash

#SBATCH -A project_2002026
#SBATCH -p test
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -J average-pathing
#SBATCH -o logs/%x-%j.out

module load pytorch
project="project_2002026"
export HF_HOME=/scratch/${project}/amanda/hf_cache
source .venv/bin/activate
metric="cosine"
jobname="full-docs-02-06-25"

full_docs_start="However, it is essential to speak to a doctor who can ease a person’s worries, determine why they have memory loss, and offer a range of treatment options."
full_docs_target="Also, not everyone on the client side may be well versed with the technicalities of SEO."
srun python embedding_extraction/average_paths.py \
                        --start="${full_docs_start}" \
                        --target="${full_docs_target}" \
                        --model="e5" \
                        --metric=$metric \
                        --save_plots=dummy\
                        --filled_indexer="/scratch/project_2002026/amanda/from-lumi/embedding-odysseia/jobs/full-docs-02-06-25/filled-indexers/IVFPQ.index" \
                        --database="/scratch/project_2002026/amanda/from-lumi/embedding-odysseia/jobs/full-docs-02-06-25/filled-indexers/IVFPQ.sqlite"


# In this tutorial we will learn about how to print Data in Tabular Format in Python



# The Regency Place Name Generator -- this generator produces quality place names geared for Regency-era stories, but could work in other time periods as well. # IN dtp
# 'However, it is essential to speak to a doctor who can ease a person’s worries, determine why they have memory loss, and offer a range of treatment options. # IN dtp
# I hope you all like it,\nits my usual mix of Crunchy Metal Rock-n-Roll and Bloody Carnage! # ID
# Understanding this will help you look for a platform that can meet those needs. # OP av
# Also, not everyone on the client side may be well versed with the technicalities of SEO. # OP av