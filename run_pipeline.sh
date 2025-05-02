#!/bin/bash
#SBATCH -A project_462000883
#SBATCH -p debug
#SBATCH --ntasks-per-node=1
##SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -J pipeline-index-full-docs-sanity
#SBATCH -o logs/%x-%j.out


module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.4
export PYTHONPATH=/scratch/project_462000883/amanda/embedding-extraction/pythonuserbase/lib/python3.10/site-packages:$PYTHONPATH


# what action to take
action=$1
case $action in
    embed|index|sanity_check)  
        # Ok
    ;;
    *)
        echo "action given poorly, give as embed, index, sanity_check, exiting"
        exit 1
esac

REGISTER=$2

jobname="full_docs"
# this will be piped in to extract.py
#data_to_embed="/scratch/project_462000883/amanda/register-data/${REGISTER}.jsonl"
data_to_embed="/scratch/project_462000353/HPLT-REGISTERS/samples-150B-by-register-xlmrl/original_corrected/eng_Latn_${REGISTER}.jsonl"
# Embedding extraction related options
model="e5"
task="STS"
data="/scratch/project_462000883/amanda/embedding-extraction/embedded-data/e5/full_docs/" #"/scratch/project_462000883/amanda/embedding-extraction/embedded-data/e5/${jobname}/${REGISTER}.pkl"  # location to save the embeddings
shard=0   #we use the first index in data (can be commaseparated list)
temporary_training_set="/scratch/project_462000883/amanda/embedding-extraction/training-data/full_docs/" #"/scratch/project_462000883/amanda/embedding-extraction/training-data/${jobname}/${REGISTER}.pkl"
threshold=0.1
split_by="truncate"
chunk_size=2500

# indexing with faiss related options
base_indexer="HNSW"
training_data="/scratch/project_462000883/amanda/embedding-extraction/training-data/${jobname}-${base_indexer}.pt"
trained_indexer="/scratch/project_462000883/amanda/embedding-extraction/trained-indexers/${jobname}-${base_indexer}.index"
filled_indexer="/scratch/project_462000883/amanda/embedding-extraction/filled-indexers/${jobname}-${base_indexer}.index"
database="/scratch/project_462000883/amanda/embedding-extraction/indexed-data/${jobname}-${base_indexer}.sqlite"
# Verbosity
debug="True"



export HF_HOME=/scratch/project_462000883/hf_cache
echo Starting at $(date +%H:%M.%S)
case $action in
    embed)  
        srun python pipeline.py \
                    --${action} \
                    --model=$model \
                    --data=$data \
                    --temp=$temporary_training_set \
                    --split_by=$split_by \
                    --threshold=$threshold \
                    --debug=$debug < $data_to_embed
    ;;
    index)
        srun python pipeline.py \
                    --${action} \
                    --base_indexer=$base_indexer \
                    --data=$data \
                    --temp=$temporary_training_set \
                    --training_data=$training_data \
                    --trained_indexer=$trained_indexer \
                    --filled_indexer=$filled_indexer \
                    --database=$database \
                    --debug=$debug
    ;;
    sanity_check)
        srun python pipeline.py \
                    --${action} \
                    --model=$model \
                    --filled_indexer=$filled_indexer \
                    --database=$database \
                    --debug=$debug
    ;;
    *)
    #should not happen
esac
echo Ending at $(date +%H:%M.%S)