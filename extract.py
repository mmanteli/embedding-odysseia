import torch
from jsonargparse import ArgumentParser
import transformers
import sys
import time
import json
import traceback
import pickle
#from transformers.pipelines.pt_utils import KeyDataset
import datetime
import datasets
import os
from sentence_transformers import SentenceTransformer


from datasets.utils.logging import disable_progress_bar      #prevent the libraries from displaying a progress bar
disable_progress_bar()

from datasets.utils.logging import set_verbosity_error
set_verbosity_error()                                        #prevent the libraries' messages from being displayed


#------------------------------- Define aliases for models here ------------------------------- #
model_name_dict = {"e5": "intfloat/multilingual-e5-large-instruct",
                   "qwen" : "Alibaba-NLP/gte-Qwen2-7B-instruct",
                   "jina" : "jinaai/jina-embeddings-v3"
}

#-------------------------------- Define possible prompts here -------------------------------- #

def get_task_def_by_task_name_and_type(task_type: str) -> str:
    if task_type in ['STS']:
        return "Retrieve semantically similar text."
    if task_type in ['Summarization']:
        return "Given a summary, retrieve other semantically similar summaries"
    if task_type in ['BitextMining']:
        return "Retrieve parallel sentences."    # this needs to be adjusted fron qwen
    if task_type in ['Retrieval']:
        return "Given a web search query, retrieve relevant passages that answer the query"
    raise ValueError(f"No instruction config for task {task_type}")

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def get_query(task_name:str, text:str) -> list:
    # get the task explanation
    task = get_task_def_by_task_name_and_type(task_name)
    # combine task description and text
    return [get_detailed_instruct(task, text)]

#--------------------------------- Chuncking of long documents --------------------------------- #

def text2dataset(line,chunk_size=5000):
    """
    turn a text into a dataset of text chunks for efficient embedding production 
    """
    txt = line["text"]
    chunks=[]
    offsets=[]
    print(line)
    for chunk_offset in range(0,len(txt),chunk_size):
        #print(f'[{chunk_offset}:{chunk_offset+chunk_size}]')
        chunks.append(txt[chunk_offset:chunk_offset+chunk_size])
        offsets.append(chunk_offset)
    return datasets.Dataset.from_dict({"text":chunks, "id":[line["id"]]*len(chunks), "register": [line["register"]]*len(chunks), "offset": offsets})

            
def transform(f, options):

    # find model with alias or full name
    model = SentenceTransformer(model_name_dict.get(options.model, options.model),trust_remote_code=True)
    if options.model in ["qwen","Alibaba-NLP/gte-Qwen2-7B-instruct"]:
        model.max_seq_length = 8192

    for idx, line in enumerate(sys.stdin):
        

        try: 
            
            j = json.loads(line)
            if options.chunk_size:
                dataset=text2dataset(j, options.chunk_size)
            else:
                dataset=datasets.Dataset.from_dict({"text":[j["text"]], "id": [j["id"]], "register": [j["register"]], "offset": [None]})
            print(dataset)
            print(dataset[0]["text"])

            dataset = dataset.map(lambda x: 
                                        {"embeddings": model.encode(
                                                                    get_query(options.task,x["text"]),  # here query to all
                                                                    convert_to_tensor=False, 
                                                                    normalize_embeddings=False
                                                                    )
                                        }
                                )
            
            pickle.dump(dataset, f)

            #first_index += tensors.shape[0]     #keeping track of the snippets ID
        except:
            traceback.print_exc()
        
            
parser = ArgumentParser(prog="extract_embeddings.py")
#parser.add_argument('--data',type=str,help="Path to dataset")
parser.add_argument('--model',type=str,help="Model name")
parser.add_argument('--save', type=str,help="Path for saving results", default=None)
parser.add_argument('--task', default="STS", choices=["STS","Summarization","BitextMining","Retrieval"], help='Task (==which query to use)')
parser.add_argument('--chunk_size',type=int,help="elements per batch", default = None)
#parser.add_argument('--total',type=int,help="How many jobs in total", default = 1)
#parser.add_argument('--rank',type=int,help="The rank of this job", default = 1)

options = parser.parse_args()

save_file = "testi.pkl" if options.save is None else options.save
assert ".pkl" in save_file, "Include a valid path with .pkl in the end"

if os.path.dirname(save_file):
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

with open(save_file, "wb") as f:
    transform(f, options)