#from sqlitedict import SqliteDict
import numpy as np
import faiss
import torch
import sys
import tqdm
import pickle
import traceback
from jsonargparse import ArgumentParser

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
    quantizer = faiss.IndexFlatL2(options.embedding_dim)
    if options.base_indexer == "IVFPQ":
        index = faiss.IndexIVFPQ(quantizer, options.embedding_dim, options.num_cells, options.num_quantizers, options.quantizer_bits)
    else:
        # using only FlatL2
        index = quantizer
    if not index.is_trained:
        training_vectors = torch.load(options.training_data)
        print(f'Training {options.base_indexer} on {len(training_vectors)} vectors.', flush=True)
        index.train(training_vectors.numpy())
        print(f'Training done. Saving to {options.trained_indexer}.', flush=True)
        index_trained = index
        faiss.write_index(index_trained, options.trained_indexer)
        return index
    return index
        
def index_w_fais(options):
    index = faiss.read_index(args.pretrained_index)
    ids, batches = make_batches(data)
    for i, b in tqdm.tqdm(zip(ids, batches)):
        index.add_with_ids(b.numpy(), i.numpy())
    index_filled = index
    faiss.write_index(index_filled, args.fill_faiss)
    print(f'Filling done, {index_filled.ntotal} vectors in index.')
    return True


if __name__ == "__main__":
    parser = ArgumentParser(prog="faissify.py")
    parser.add_argument('--base_indexer',type=str,help="Indexer name, e.g. IVFPQ")
    parser.add_argument('--trained_indexer',type=str,help="Path to save/load the indexer.")
    parser.add_argument('--data', type=str, help="Path to data")
    parser.add_argument('--training_data', type=str, help="Path to load or save training data.")
    parser.add_argument('--embedding_dim', type=int,help="Embedding dimension", default=1024)
    parser.add_argument('--num_cells',type=int,help="Number of Voronoi cells in IVF", default = 1024)
    parser.add_argument('--num_quantizers', type=int, help="Number of quantizer in PQ", default = 64)
    parser.add_argument('--quantizer_bits', type=int, help="How many bits used to save quantizer values", default = 8)
    parser.add_argument('--debug', type=bool, default=False, help="Verbosity etc.")


