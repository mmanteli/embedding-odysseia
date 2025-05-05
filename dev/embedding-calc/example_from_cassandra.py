import torch
import argparse
import transformers
import sys
import time
import json
import traceback
import pickle
from transformers.pipelines.pt_utils import KeyDataset
import datetime
import datasets

from datasets.utils.logging import disable_progress_bar      #prevent the libraries from displaying a progress bar
disable_progress_bar()

from datasets.utils.logging import set_verbosity_error
set_verbosity_error()                                        #prevent the libraries' messages from being displayed

def book2dataset(txt,chunk_size=100000):                          #turn a book into a dataset of text chunks for efficient embedding production 
    chunks=[]
    for chunk_offset in range(0,len(txt),chunk_size):
        chunks.append(txt[chunk_offset:chunk_offset+chunk_size])
    return datasets.Dataset.from_dict({"text":chunks})

def tokenize(ex):
    texts,offsets=sliding_window(ex["text"])
    return {"texts":texts,"offset":offsets}
        
def sliding_window(text):                                    #cut the text in overlapping snippets 
        vectors = []
        off_vectors = []
        try :     
            tkns = tokenizer(text,max_length = 100,stride = 30,truncation=True,add_special_tokens=False,return_overflowing_tokens=True, return_offsets_mapping=True)
            token_strings = tkns["offset_mapping"]
            for list in token_strings:
                window_start = list[0][0]
                window_end = list[-1][1]
                vectors.append(text[window_start:window_end])
                off_vectors.append((window_start, window_end))
            return(vectors, off_vectors)
        
        except:
            traceback.print_exc()
            
def transform(f,chunk_size=100000):
    
    first_index = 100000000000 * args.rank        #custom ID for every snippet of the dataset
    
    for idx, line in enumerate(sys.stdin):
        
        if idx%args.total!=args.rank:            #parallelization 
            continue

        try : 
                book = json.loads(line)
                book_chunks_dataset=book2dataset(book["text"],chunk_size)
                book_chunks_dataset=book_chunks_dataset.map(tokenize,num_proc=8)

                dataset=[]
                offsets=[]
                for chunk_idx,t_chunk in enumerate(book_chunks_dataset):

                    offsets.extend([(off_idx[0]+chunk_size*chunk_idx, off_idx[1]+chunk_size*chunk_idx) for off_idx in t_chunk["offset"]])
                    dataset.extend(t_chunk["texts"])

                dataset, offsets = sliding_window(book["text"])

                dset=[{"text":t} for t in dataset]
                dset=KeyDataset(dset,"text")
                emb=p(dataset, batch_size=(args.batchsize), truncation="only_first")
                emb_pool=[(torch.mean(elem,axis=1)).squeeze(0) for elem in emb]
                tensors = torch.vstack(emb_pool)  

                pickle.dump(({"book name": book["url"], "offsets": offsets, "index of first":first_index}, tensors), f)

                first_index += tensors.shape[0]     #keeping track of the snippets ID


        except:
            traceback.print_exc()
        
            
parser = argparse.ArgumentParser()
parser.add_argument('--batchsize',type=int,help="elements per batch", default = 200)
parser.add_argument('--total',type=int,help="How many jobs in total")
parser.add_argument('--rank',type=int,help="The rank of this job")
args = parser.parse_args()

p=transformers.pipeline(task="feature-extraction",model="sentence-transformers/paraphrase-xlm-r-multilingual-v1",return_tensors=True, device =0) #device = 0 to run on GPU

tokenizer=transformers.AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-xlm-r-multilingual-v1")

document = "/scratch/project_2005072/cassandra/tensors/tensors_ecco/ecco_tensors_"+str(args.rank)+".pickle"

with open(document, "wb") as f:
    transform(f)