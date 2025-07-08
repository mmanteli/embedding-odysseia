#!/bin/bash

#SBATCH -A project_2002026
#SBATCH -p small
#SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -J average-pathing
#SBATCH -o logs/%x-%j.out

module load pytorch
project="project_2002026"
export HF_HOME=/scratch/${project}/amanda/hf_cache
source .venv/bin/activate
metric="cosine"
jobname="full-docs-02-06-25"
a=$1
b=$2
a1=$(echo $a | tr "." "_")
b1=$(echo $b | tr "." "_")

full_docs_start="However, it is essential to speak to a doctor who can ease a person’s worries, determine why they have memory loss, and offer a range of treatment options."
full_docs_target="Also, not everyone on the client side may be well versed with the technicalities of SEO."
#full_docs_start="You can include references like 1 John 4:19 or even Eph 2:8-9 on your site. Reftagger automatically tags the reference and creates a tooltip that appears when a reader hovers over it.\nthe passage on BibleGateway.com as well.\nJeremiah 29:11\nRom 8:28\nProv 3:5-6\n1 Corinthians 13:1-13\nSave yourself the trouble"
#full_docs_start="In addition to the regular business of Council, the Town holds public meetings about development applications, by-laws, plans, zoning changes and more.\nThe Town of Caledon is offering a hybrid meeting model offering many ways for you to engage with Council. To participate in-person or virtually, complete the participation form by registering here.\nUpcoming meetings\nNovember 28, 2022 | 7 p.m.\n- 12304 Heart Lake Road – Phase 2\n- 0 King Street\nPast meetings\nHow to Participate\nMembers of the public may participate and learn more about proposed applications and can provide direct input."
#full_docs_target="by: Bonaldo\n$ 1,520.00\nFun Bookcase is a contemporary modular bookshelf design fashioned with irregular divisions that create a pleasing effect of movement. The Fun bookcase is made of MDF wood lacquered in matte colors black, white or capuccino. Several modules available for compositions. Select from 2, 3 or 5 shelf levels. This piece of living room furniture be positioned as a room divider or along walls, depending on depth. Side by side modularity.\nDimensions:\nWidth: 47W (Inches)\nDepths: 11D;15D\nHeights: 32H; 47H; 79H\nBonaldo is among the finest design houses in Italy. They make products that help people create and complement contemporary themes. The living collection feature minimalistic style and virtually everything else that people have come to expect from modernistic furniture. Since 1936, the group has been transforming ideas into designs and making items that can successfully interpret the requirements of the modern world, generating excitement at first sight. They focus on identifying and eliminating even the smallest blemish or fault which is the means for quality. Communicates designs with style, color and dynamism for materials while giving attention and care for every tiny detail. Take pride in being a first-hand witness to the beauty and simple elegance of their products. The label says it all when it comes to dining and living room tables and chairs, bookcase, sideboards, beds of all sizes - even for kids, their armchairs and sofas.\nView Specifications\nLead time: 9 to 12 weeks\n$ 3,077.00\nContemporary Modern Lacquered/Wood Coffee Table The Bench Coffee Table designed by Giuseppe Bavuso is a beautiful addition to any home or public lounge that requires a modern touch. This coffee...\n$ 1,829.00\nUnique Modern Coffee Table Design. White Stone/Painted Top. Bloom Maxi is a modern coffee table designed by Giuseppe Bavuso which features a unique shaped tabletop on a chrome-plated or painted...\n$ 1,540.00\nUltra Modern Low Coffee Table. Italy Design Contemporary Lacquered Coffee Table. Sleek, simple and yet distinctive, the Daytona Coffee Table can be matched with the Club Lounge Chair and Side..."
srun python embedding_extraction/average_paths.py \
                        --start="${full_docs_start}" \
                        --target="${full_docs_target}" \
                        --model="e5" \
                        --alpha=$a \
                        --beta=$b \
                        --metric=$metric \
                        --save_plots="/scratch/project_2002026/amanda/from-lumi/embedding-odysseia/averaging-results/results3/fig-${a1}-${b1}"\
                        --filled_indexer="/scratch/project_2002026/amanda/from-lumi/embedding-odysseia/jobs/full-docs-02-06-25/filled-indexers/IVFPQ.index" \
                        --database="/scratch/project_2002026/amanda/from-lumi/embedding-odysseia/jobs/full-docs-02-06-25/filled-indexers/IVFPQ.sqlite"


# In this tutorial we will learn about how to print Data in Tabular Format in Python



# The Regency Place Name Generator -- this generator produces quality place names geared for Regency-era stories, but could work in other time periods as well. # IN dtp
# 'However, it is essential to speak to a doctor who can ease a person’s worries, determine why they have memory loss, and offer a range of treatment options. # IN dtp
# I hope you all like it,\nits my usual mix of Crunchy Metal Rock-n-Roll and Bloody Carnage! # ID
# Understanding this will help you look for a platform that can meet those needs. # OP av
# Also, not everyone on the client side may be well versed with the technicalities of SEO. # OP av