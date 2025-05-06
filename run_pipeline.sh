#!/bin/bash
project="project_462000883"

module_setup="module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.4
export PYTHONPATH=/scratch/project_462000883/amanda/embedding-odysseia/pythonuserbase/lib/python3.10/site-packages:$PYTHONPATH
export HF_HOME=/scratch/project_462000883/hf_cache"


# This script used to run embedding, indexing and a sanity check in one, to make argument handling consistent.
# use as 
#
# ./run_pipeline.sh [embed|index|sanity_check]
#
# This starts batchjobs for pipeline.py which redirects to extract.py (embed), faissify.py (index) or sanity_check.py (sanity).
# (run "chmod +x run_pipeline.sh" if it complains about rights)
# As we want the embedding calculation to be in parallel, give the data in batches,
# those are piped to extract.py.
# I.e. have your data in medium sized shards.
# modify options in this file, or make a copy, or don't, I'm just text on a screen.
# Read the comments in this file (or documentation, if it exists) carefully!


# MOST IMPORTANT PARAMETERS
# this will affect saving paths
jobname="embed-and-index" # you can for example add this-> $(date +%d-%m-%y) to get a date in the name, if everything is run on the same day
# this will be piped in to extract.py, can be path, then all .jsonl's in the path will be piped in parallel jobs.
data_to_embed="/scratch/project_462000883/amanda/register-data/"
path_prefix_for_results="/scratch/project_462000883/amanda/embedding-extraction/jobs/${jobname}"

# what action to take
action=$1
case $action in
    embed|index|sanity_check)  
        # Ok
    ;;
    *)
        echo "action given poorly, give as embed, index, sanity_check. Exiting"
        exit 1
esac


# Embedding extraction related options
model="e5"
task="STS"
data="${path_prefix_for_results}/embedded-data/${model}/" # location to save the embeddings, and read in indexing
temporary_training_set="${path_prefix_for_results}/training-data/" # location for temp training data files, read in indexing, so that we dont have to go through the data twice
data_suffix=""  # suffix for saving the data. Applied to both above!! See loop in "embed" below, where we assing this !!!!
threshold=0.1   # which fraction of data is selected for training the indexer, usually 0.1 is more than enough. 
# faissify.py will complain if it is too little, and the indexer will not work. You don't have to re-run the embedding step, faissify.py can create its own training data if temporary training set does not exist.
split_by="truncate"   # this is what to use to divide long documents to chunks. Truncate: none, just beginning of file, sentences: find sentences using nltk, words/chars: select number of units.
chunk_size=2500  # this is the number of units chosen, for example if sentence is more than 2500 chars long, this can be used to truncate. 2500 char ~= 512 tokens

# indexing with faiss + sanity check related options
base_indexer="IVFPQ"    # indexer type. IVFPQ is fast but not that accurate, HNSW is memory hungry and slow but nice, Flat2D is best but not for large data.
training_data="${path_prefix_for_results}/training-data/${base_indexer}_training_data.pt"   # if no temp training data, this is created. If temp training data, it is concatenated, shuffled and saved here.
trained_indexer="${path_prefix_for_results}/trained-indexers/${base_indexer}.index"  # save trained indexer here (mainly in case filling the index crashes)
filled_indexer="${path_prefix_for_results}/filled-indexers/${base_indexer}.index" # save filled indexer here
database="${path_prefix_for_results}/filled-indexers/${base_indexer}.sqlite" # save corresponding sql database here
# Verbosity
debug="True"



case $action in
    embed)
        if [[ -d $data_to_embed ]]; then   # if a dir, loop over files
            for filename in $data_to_embed*.jsonl; do
            # define the data suffix:
            data_suffix=$(basename "$filename" .jsonl)   # basename without the extension
            CMD="srun python pipeline.py \
                        --${action} \
                        --model=$model \
                        --data=$data \
                        --temp=$temporary_training_set \
                        --data_suffix=$data_suffix \
                        --split_by=$split_by \
                        --threshold=$threshold \
                        --debug=$debug < $filename"
            sbatch --job-name=embed \
                --account=$project \
                --output=${path_prefix_for_results}/logs/%x-%j.out \
                --time=01:30:00 \
                --partition=small-g \
                --nodes=1 \
                --ntasks=1 \
                --gpus-per-node=1 \
                --cpus-per-task=4 \
                --mem=20G <<EOF
#!/bin/bash
echo "Starting: \$(date)"
echo "Running embedding..."
echo $CMD

$module_setup
$CMD
echo "Ending: \$(date)"
EOF
            done
        elif [[ -f $data_to_embed ]]; then   # if one file
            if [[ $data_to_embed != *.jsonl ]]; then
                echo "File to embed given incorrectly. Give as dir containing jsonl's, or one jsonl file"
                exit 1
            fi
            data_suffix=$(basename "$filename" .jsonl)   # basename without the extension
            CMD="srun python pipeline.py \
                        --${action} \
                        --model=$model \
                        --data=$data \
                        --temp=$temporary_training_set \
                        --data_suffix=$data_suffix \
                        --split_by=$split_by \
                        --threshold=$threshold \
                        --debug=$debug < $data_to_embed"
            sbatch --job-name=embed \
                --account=$project \
                --output=${path_prefix_for_results}/logs/%x-%j.out \
                --time=02:30:00 \
                --partition=small-g \
                --nodes=1 \
                --ntasks=1 \
                --gpus-per-node=1 \
                --cpus-per-task=4 \
                --mem=20G <<EOF
#!/bin/bash
echo "Starting: \$(date)"
echo "Running embedding..."
echo $CMD

$module_setup
$CMD
echo "Ending: \$(date)"
EOF
        else
                echo "Embed option given with invalid data to embed. Give as dir containing jsonl's, or one jsonl file."
        fi
    ;;
    index)
        CMD="srun python pipeline.py \
                    --${action} \
                    --base_indexer=$base_indexer \
                    --data=$data \
                    --temp=$temporary_training_set \
                    --training_data=$training_data \
                    --trained_indexer=$trained_indexer \
                    --filled_indexer=$filled_indexer \
                    --database=$database \
                    --debug=$debug"
        sbatch --job-name=index \
               --account=$project \
               --output=${path_prefix_for_results}/logs/%x-%j.out \
               --time=02:30:00 \
               --partition=small \
               --nodes=1 \
               --ntasks=1 \
               --cpus-per-task=4 \
               --mem=20G <<EOF
#!/bin/bash
echo "Starting: \$(date)"
echo "Running indexing..."
echo $CMD

$module_setup
$CMD
echo "Ending: \$(date)"
EOF
    ;;
    sanity_check)
        CMD="srun python pipeline.py \
                    --${action} \
                    --model=$model \
                    --filled_indexer=$filled_indexer \
                    --database=$database \
                    --debug=$debug"
        sbatch --job-name=sanity \
               --account=$project \
               --output=${path_prefix_for_results}/logs/%x-%j.out \
               --time=00:30:00 \
               --partition=debug \
               --nodes=1 \
               --ntasks=1 \
               --cpus-per-task=4 \
               --mem=20G <<EOF
#!/bin/bash
echo "Starting: \$(date)"
echo "Running sanity check..."
echo $CMD

$module_setup
$CMD
echo "Ending: \$(date)"
EOF
    ;;
    *)
        echo "This should not happen!"
esac
