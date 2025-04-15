#from sqlitedict import SqliteDict
import numpy as np
import faiss
import torch
import sys
import tqdm
import pickle
import traceback
from jsonargparse import ArgumentParser
import pathlib

def yield_from_pickle(f_name):
    with open(f_name,"rb") as f:
        while True:
            try:
                dicts=pickle.load(f)
                yield dicts
            except EOFError:
                break

def make_training_sample(options, fraction = 0.1):
    """
    Sample fraction (==0.1) of the data for training.
    Save the data to options.training_data, return True on success
    """
    if options.debug: print(f'Making a training set with fraction {fraction}', flush=True)
    training_embeddings = []
    try:
        for beet in yield_from_pickle(options.data):
            for b in beet:
                if np.random.random() < fraction:
                    training_embeddings.append(b["embeddings"])
        torch.save(training_embeddings, options.training_data)
        return True
    except:
        traceback.print_exc()
        return False


def train_faiss(options):
    """
    Trains Faiss index if it is untrained. Some indexers do not need training.
    Returns the trained index, also saves it to options.trained_indexer.
    """
    quantizer = faiss.IndexFlatL2(options.embedding_dim)
    if options.base_indexer == "IVFPQ":
        index = faiss.IndexIVFPQ(quantizer, options.embedding_dim, options.num_cells, options.num_quantizers, options.quantizer_bits)
    else:
        # using only FlatL2
        if debug: print("Using FlatL2 as the full indexer.", flush=True)
        index = quantizer
    if not index.is_trained:
        training_vectors = torch.load(options.training_data)
        if options.debug: print(f'Training {options.base_indexer} on {len(training_vectors)} vectors.', flush=True)
        index.train(training_vectors.numpy())
        if options.debug: print(f'Training done. Saving to {options.trained_indexer}.', flush=True)
        index_trained = index
        faiss.write_index(index_trained, options.trained_indexer)
        return index
    return index
        

def extract_embedding_and_id(d):
    return d["embeddings"], d["id"]

def index_w_fais(options, index=None):
    if index is None:
        if debug: print(f"Reading {options.trained_indexer}...", flush=True)
        index = faiss.read_index(options.trained_indexer)
    # collect batches
    emb_to_index = []
    id_to_index = []
    for i, beet in enumerate(yield_from_pickle(options.data)):
        emb_, id_ = extract_embedding_and_id(d)
        emb_to_index.append(emb_)
        id_to_index.append(id_)
        if i%1000 == 0:  # for every batch
            index.add_with_ids(emb_to_index.numpy(), id_to_index.numpy())
            emb_to_index = []
            id_to_index = []
    # tail values
    index.add_with_ids(emb_to_index.numpy(), id_to_index.numpy())
    index_filled = index
    faiss.write_index(index_filled, options.filled_indexer)
    if options.debug: print(f'Filling done, {index_filled.ntotal} vectors in index.', flush=True)
    return True


def run(options):
    # check if the training data is already done
    if not pathlib.Path(options.training_data).is_file():
        data_exists = make_training_sample(options)
        if not data_exists:
            print("Could not construct training data, see error above.", flush=True)
            exit(1)
    indexer = None
    if not pathlib.Path(options.trained_indexer).is_file():
        if options.debug: print("Did not find a trained indexer, training...")
        indexer = train_faiss(options)
    if indexer:
        index_w_fais(options, index=indexer)
    else:
        index_w_fais(options)
    
    
    

if __name__ == "__main__":
    parser = ArgumentParser(prog="faissify.py")
    parser.add_argument('--base_indexer',type=str,help="Indexer name, e.g. IVFPQ")
    parser.add_argument('--trained_indexer',type=str,help="Path to save/load the indexer.")
    parser.add_argument('--filled_indexer',type=str,help="Path to fill the indexer.")
    parser.add_argument('--data', type=str, help="Path to data")
    parser.add_argument('--training_data', type=str, help="Path to load or save training data.")
    parser.add_argument('--embedding_dim', type=int,help="Embedding dimension", default=1024)
    parser.add_argument('--num_cells',type=int,help="Number of Voronoi cells in IVF", default = 1024)
    parser.add_argument('--num_quantizers', type=int, help="Number of quantizer in PQ", default = 64)
    parser.add_argument('--quantizer_bits', type=int, help="How many bits used to save quantizer values", default = 8)
    parser.add_argument('--debug', type=bool, default=True, help="Verbosity etc.")

    # option checks
    assert ".pkl" in options.data or ".pickle" in options.data, "Give training data in pickled json format."
    if options.training_data:
        assert ".pt" in options.training_data, "Training data does not have .pt extension."
    run(options)
