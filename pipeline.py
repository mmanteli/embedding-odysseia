from jsonargparse import ArgumentParser
import sys
from extract import transform
from faissify import make_training_sample, train_faiss, index_w_fais
from sanity_check import get_NN
import os
import glob
import pathlib


### Master arguments
parser = ArgumentParser(prog="Embedding extraction and indexing pipeline")
# what action to take
parser.add_argument('--embed', action="store_true")
parser.add_argument('--index', action="store_true")
parser.add_argument('--sanity_check', action="store_true")
# Embedding extraction related options
parser.add_argument('--model',type=str, help="Model name")
parser.add_argument('--task', default="STS", choices=["STS","Summarization","BitextMining","Retrieval"], help='Task (==which query to use)')
parser.add_argument('--data','--embedded', '--save', type=str, help="Path to save embedings in .pkl format, can only be dir, then set data_suffix.", default=None)
parser.add_argument('--data_suffix', type=str, default=None, help="See --data.")
parser.add_argument('--batch_size', '--batchsize', type=int,help="Document batch size for embeddings", default = 500) 
parser.add_argument('--model_batch_size', type=int, help="Model inner batch size", default=32)  # tested to be the fastest out of 32 64 128
parser.add_argument('--split_by', default="sentences", choices=["tokens", "sentences", "chars", "words", "truncate"], help='What to use for splitting too long texts, truncate=nothing')
parser.add_argument('--chunk_size', type=int, help="How many units (tokens, words, chars) per batch. For sentences, splits up by char, for tokens, automatically set to model_max_len -2", default = None)
parser.add_argument('--overlap', '--context_overlap', type=int, help="How much overlap per segment", default = None)
parser.add_argument('--temporary_training_set', '--temp', type=str, help='sample training data to temporary .pkl, read to .pt in indexing')
parser.add_argument('--threshold', type=float, default=0.1, help='Sample fraction for temp training data')
# indexing with faiss related options
parser.add_argument('--base_indexer',type=str,help="Faiss indexer type, e.g. IVFPQ")
parser.add_argument('--training_data', type=str, help="Path to load or save training data, .pt.")
parser.add_argument('--trained_indexer',type=str,help="Path to save/load the indexer. Extension: .index")
parser.add_argument('--filled_indexer',type=str,help="Path to fill the indexer. Extension: .index")
parser.add_argument('--database', type=str, help="Where to save the indexed data in text format, .sqlite.")
parser.add_argument('--embedding_dim', type=int,help="Embedding dimension", default=1024)
parser.add_argument('--num_cells',type=int,help="Number of Voronoi cells in IVF", default = 1024)
parser.add_argument('--num_quantizers', type=int, help="Number of quantizer in PQ", default = 64)
parser.add_argument('--quantizer_bits', type=int, help="How many bits used to save quantizer values", default = 8)
parser.add_argument('--graph_connections', type=int, help="How many connections in HSNW", default = 64)
parser.add_argument('--ef_construction', type=int, help="How many layers for graph construction", default = 128)
parser.add_argument('--ef_search', type=int, help="How layers for graph searc", default = 128)  
# Verbosity
parser.add_argument('--debug', type=bool, default=False, help="Verbosity etc.")

options = parser.parse_args()





# check that we have a task to do
if not any([options.embed, options.index, options.sanity_check]):
    print("No action given to the program. Give '--embed', '--index', and/or '--sanity_check' as flag(s).")
    exit(1)

if options.embed:
    assert options.model is not None, "Give --model to embed with"
    assert options.data is not None, "Give --data path to save embedding results to"
    assert options.data_suffix is not None or ".pkl" in options.data, f"Give the saving location for embeddings as dir (--data={options.data}) and suffix (--data_suffix={options.data_suffix}), or --data as one .pkl file"
    # check format
    if options.data_suffix and ".pkl" not in options.data_suffix:
        options.data_suffix += ".pkl"
    if options.split_by == "sentences":
        options.chunk_size=2500   # about model sequence length

if options.index:
    assert (options.base_indexer is not None and options.training_data is not None) or options.trained_indexer is not None, "Give either base indexer and training data or trained indexer to index."
    assert options.database is not None, "Give a --database to save results to to index"
    assert ".sqlite" in options.database, "Give valid --database path to an sqlite database (.sqlite)"
    # parse --data for indexing (here you can give it as a more nuanced format):
    # handle data paths, which can be given as a comma-separated list:
    if options.data is None:
        options.data = False # not needed every time, e.g. if only training
    elif "," in options.data:
        # It's a list, parse with split
        options.data = [d for d in options.data.split(",")]
    elif os.path.isdir(options.data):
        # It's a directory, find all .pkl files
        options.data = glob.glob(os.path.join(options.data, '*.pkl'))
    else:
        pass
    # Same for training data
    if options.temporary_training_set is None:
        options.temporary_training_set = False 
    elif "," in options.temporary_training_set:
        # It's a list, parse with split
        options.temporary_training_set = [d for d in options.temporary_training_set.split(",")]
    elif os.path.isdir(options.temporary_training_set):
        # It's a directory, find all .pkl files
        options.temporary_training_set = glob.glob(os.path.join(options.temporary_training_set, '*.pkl'))
    else:
        pass

if options.sanity_check:
    assert options.filled_indexer is not None and options.database is not None, "Give filled indexer and databse to do the sanity check"
    # See if database is in correct format
    assert ".sqlite" in options.database, "Give valid path to an sqlite database (.sqlite)"




print(options, flush=True)




### EMBEDDING
if options.embed:
    # If create temp training set at the same time:
    if options.temporary_training_set:
        if options.data_suffix:
            save_embed_file = os.path.join(options.data, options.data_suffix)
            temp_train_file = os.path.join(options.temporary_training_set, options.data_suffix)
        else:
            save_embed_file = options.data
            temp_train_file = options.temporary_training_set
        # assert that extension is correct
        assert save_embed_file[-4:]==".pkl" and temp_train_file[-4:]==".pkl" and "," not in save_embed_file and "," not in temp_train_file, f"--data or --temporary_training_set read incorrectly save = {save_embed_file}, temp = {temp_train_file} "
        # make directories if needed
        if os.path.dirname(save_embed_file):
            os.makedirs(os.path.dirname(save_embed_file), exist_ok=True)
        if os.path.dirname(temp_train_file):
            os.makedirs(os.path.dirname(temp_train_file), exist_ok=True)
        if options.debug: print(f"Running embedding calculations, saving to {save_embed_file}, and creating temp training set to {temp_train_file}", flush=True)
        with open(save_embed_file, "wb") as f, open(temp_train_file, "wb") as f_train:
            transform(f, options, f_train=f_train)
    # If no training data created in this step
    else:
        if options.data_suffix:
            save_embed_file = os.path.join(options.data, options.data_suffix)
        else:
            save_embed_file = options.data
        # assert that extension is correct and that list is only given with a shard
        assert save_embed_file[-4:]==".pkl" and "," not in save_embed_file, f"--data or --temporary_training_set read incorrectly save = {save_embed_file}, temp = {temp_train_file} "
        # make directories if needed
        if os.path.dirname(save_embed_file):
            os.makedirs(os.path.dirname(save_embed_file), exist_ok=True)
        if options.debug: print(f"Running embedding calculations, saving to {save_embed_file}, not creating temp training set.", flush=True)
        with open(save_embed_file, "wb") as f:
            transform(f, options)


### INDEXING
if options.index:
    # check if the trained indexer already exists:
    indexer = None   # this is so we can differentiate if training was done in this step or earlier, we do not have to load the index if we just trained it
    os.makedirs(os.path.dirname(options.trained_indexer), exist_ok=True)
    os.makedirs(os.path.dirname(options.filled_indexer), exist_ok=True)
    os.makedirs(os.path.dirname(options.database), exist_ok=True)
    if not pathlib.Path(options.trained_indexer).is_file():
        # check if training data exists in .pytorch format:
        if not pathlib.Path(options.training_data).is_file():
            if options.debug: print("Did not find training data in pytorch format")
            if options.temporary_training_set:
                if options.debug: print(f"Creating from temp training data {options.temporary_training_set}")
                training_sample_exists = make_training_sample(options.temporary_training_set, options.training_data, debug=options.debug, fraction = 1) # NOTE full data here
            else:
                training_sample_exists = make_training_sample(options.data, options.training_data, debug=options.debug, fraction=options.threshold)
            assert training_sample_exists, f"Could not make training data for reasons that should have been reported above"
        if options.debug: print("Now moving to training faiss index", flush=True)
        indexer = train_faiss(options)
    if options.debug: print("Now moving to indexing", flush=True)
    index_w_fais(options, index = indexer)

### Sanity
if options.sanity_check:
    if options.debug: print("In sanity check", flush=True)
    get_NN(options)

