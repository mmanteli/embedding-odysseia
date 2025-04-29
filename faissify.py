from sqlitedict import SqliteDict
import numpy as np
import faiss
import torch
import sys
import tqdm
import pickle
import traceback
from jsonargparse import ArgumentParser
import pathlib

def yield_from_pickle(f_names):
    """
    Generator to read from a list of pickled files (or one pickle file)
    """
    if isinstance(f_names, str):
        f_names = [f_names]
    for f_name in f_names:
        with open(f_name,"rb") as f:
            while True:
                try:
                    dicts=pickle.load(f)
                    if type(dicts) == list:   # some older sets are saved this way
                        for d in dicts:
                            yield d
                    else:
                        yield dicts
                except EOFError:
                    break   # this breaks while

def make_training_sample(data, training_data, debug=False, fraction = 0.1):
    """
    Sample fraction (==0.1) of the data for training.
    Save the data to training_data, return True on success.
    """
    if debug: print(f'Making a training set from {data} with fraction {fraction}, saving to {training_data}', flush=True)
    training_embeddings = []
    try:
        for beet in yield_from_pickle(data):
            if np.random.random() < fraction:
                training_embeddings.append(beet["embeddings"])
        training_embeddings = np.vstack([e for e in training_embeddings])
        torch.save(training_embeddings, training_data)
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
    elif options.base_indexer == "HNSW":
        M = options.graph_connections # connections
        d = options.embedding_dim
        index = faiss.IndexHNSWFlat(d,M)
        index.hnsw.efConstruction = options.ef_construction
        index.hnsw.efSearch = options.ef_search
    else:
        # using only FlatL2
        if options.debug: print("Using FlatL2 as the full indexer.", flush=True)
        index = quantizer
    if not index.is_trained or options.base_indexer != "HNSW":
        training_vectors = torch.load(options.training_data)
        if options.debug: print(f'Training {options.base_indexer} on {len(training_vectors)} vectors.', flush=True)
        index.train(training_vectors)
        if options.debug: print(f'Training done. Saving to {options.trained_indexer}.', flush=True)
        index_trained = index
        faiss.write_index(index_trained, options.trained_indexer)
        return index
    else:
        if options.debug: print("This index does not require training.", flush=True)
        return index
        

def index_w_fais(options, index=None):
    db = SqliteDict(options.database)
    if index is None:
        if options.debug: print(f"Reading {options.trained_indexer}...", flush=True)
        index = faiss.read_index(options.trained_indexer)
    # collect batches
    emb_to_index = []
    id_to_index = []
    for i, beet in enumerate(yield_from_pickle(options.data)):
        emb = beet["embeddings"]
        if type(emb) != np.ndarray:
            print(type(emb))
            print(f'error2 at {i}', flush=True)
        emb_to_index.append(emb)
        id_to_index.append(i)
        db[i] = beet
        if (i+1)%1000 == 0:  # for every batch
            E, I = np.vstack([e for e in emb_to_index]), np.array(id_to_index)
            if options.base_indexer == "HNSW":
                index.add(E)  # Here we need to trust that they match
            else:
                index.add_with_ids(E, I)
            db.commit()
            # reinit
            emb_to_index = []
            id_to_index = []
    # tail values
    E, I = np.vstack([e for e in emb_to_index]), np.array(id_to_index)
    if options.base_indexer == "HNSW":
        index.add(E)  # Here we need to trust that they match
    else:
        index.add_with_ids(E, I)
    db.commit()
    if options.debug: print("Filling done, saving filled index", flush=True)
    index_filled = index
    faiss.write_index(index_filled, options.filled_indexer)
    if options.debug: print(f'{index_filled.ntotal} vectors in index.', flush=True)
    return True


def run(options):
    # check if the training data is already done
    if not pathlib.Path(options.training_data).is_file():
        if options.debug: print("Did not find training data, making it from options.data.")
        #data_exists = make_training_sample(options)   # removed the dependency of options here
        data_exists = make_training_sample(options.data, options.training_data, debug=options.debug)
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
    parser.add_argument('--trained_indexer',type=str,help="Path to save/load the indexer .index.")
    parser.add_argument('--filled_indexer',type=str,help="Path to fill the indexer.")
    parser.add_argument('--data', type=str, help="Path to data")
    parser.add_argument('--database', type=str, help="Where to save the indexed data in text format, .sqlite.")
    parser.add_argument('--training_data', type=str, help="Path to load or save training data, .pt.")
    parser.add_argument('--embedding_dim', type=int,help="Embedding dimension", default=1024)
    parser.add_argument('--num_cells',type=int,help="Number of Voronoi cells in IVF", default = 1024)
    parser.add_argument('--num_quantizers', type=int, help="Number of quantizer in PQ", default = 64)
    parser.add_argument('--quantizer_bits', type=int, help="How many bits used to save quantizer values", default = 8)
    parser.add_argument('--graph_connections', type=int, help="How many connections in HSNW", default = 64)
    parser.add_argument('--ef_construction', type=int, help="How many layers for graph construction", default = 128)
    parser.add_argument('--ef_search', type=int, help="How layers for graph searc", default = 128)  
    parser.add_argument('--debug', type=bool, default=True, help="Verbosity etc.")

    options = parser.parse_args()
    print(options, flush=True)
    # option checks
    options.data = [d for d in options.data.split(",")]
    assert ".pkl" in options.data[0] or ".pickle" in options.data[0], "Give data in pickled json format."
    if options.training_data:
        assert ".pt" in options.training_data, "Training data does not have .pt extension."
    assert ".sqlite" in options.database, "Give valid path to an sqlite database (.sqlite)"
    run(options)
