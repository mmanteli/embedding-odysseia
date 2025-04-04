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

def text2chunks(line,chunk_size=5000):
    """
    turn a text into chunked dict for further processing 
    """
    txt = line["text"]
    chunks=[]
    offsets=[]
    for chunk_offset in range(0,len(txt),chunk_size):
        chunks.append(txt[chunk_offset:chunk_offset+chunk_size])
        offsets.append(chunk_offset)
    return {"text":chunks, "id":[line["id"]]*len(chunks), "register": [line["register"]]*len(chunks), "offset": offsets}
            
def transform_with_dict_and_batch(f, model, options):

    raw_data = dict(text=[], id=[], register=[], offset=[])
    for idx, line in enumerate(sys.stdin):
        print(f'In idx {idx}')
        try: 
            j = json.loads(line)
            print("HERE\n",j)
            if options.chunk_size:
                chunked = text2chunks(j, options.chunk_size)
                for key in raw_data.keys():
                    for k in chunked[key]:
                        raw_data[key].append(k)
                
            else:
                for key in raw_data.keys():
                    raw_data[key].append(j.get(key, None))
        except:
            traceback.print_exc()
        print(raw_data)
        print(len(raw_data["text"]))

        if len(raw_data["id"]) >= options.batch_size:
            input_texts = [get_query(options.task, t)[0] for t in raw_data["text"]]
            print("INPUT")
            print(len(input_texts))
            emb = model.encode(input_texts, convert_to_tensor=False, normalize_embeddings=False)
            raw_data["embeddings"] = emb

            dataset = datasets.Dataset.from_dict(raw_data)
            #pickle.dump(dataset, f)
            raw_data=dict(text=[], id=[], register=[], offset=[])# re-init


def transform_with_dict_and_map(f, model, options):

    raw_data = dict(text=[], id=[], register=[], offset=[])
    for idx, line in enumerate(sys.stdin):
        print(f'In idx {idx}')
        try: 
            j = json.loads(line)
            if options.chunk_size:
                chunked = text2chunks(j, options.chunk_size)
                for key in raw_data.keys():
                    for k in chunked[key]:
                        raw_data[key].append(k)
                
            else:
                for key in raw_data.keys():
                    raw_data[key].append(j[key])
        except:
            traceback.print_exc()
        print(raw_data)
        print(len(raw_data["text"]))

        if len(raw_data["id"]) >= options.batch_size:
            dataset = datasets.Dataset.from_dict(raw_data)

            dataset = dataset.map(lambda x: 
                                        {"embeddings": model.encode(
                                                                    get_query(options.task,x["text"]),  # here query to all
                                                                    convert_to_tensor=False, 
                                                                    normalize_embeddings=False
                                                                    )
                                        }
                                )
            
            #pickle.dump(dataset, f)
            raw_data=dict(text=[], id=[], register=[], offset=[])# re-init

def transform_with_pandas(f, model, data, options):

    df = pd.read_json(data, lines=True)
    df = df.head(10)
    print(df.columns)


    embeddings = []
    if True:
        for index, row in df.iterrows():
            query = get_query(options.task, row["text"])
            input_texts = query + [] # documents
            embeddings.append(model.encode(input_texts, convert_to_tensor=False, normalize_embeddings=False)[0])
    else:
        for index, row in df.iterrows():
            query = get_query(options.task, row["text"])
            input_texts = query + [] # documents
            tokenizer = AutoTokenizer.from_pretrained(model_name_dict[options.model])
            tok = tokenizer(input_texts)
            print(tok)
            emb = model.forward(input=tok, convert_to_tensor=False, normalize_embeddings=False)
            embeddings.append(emb[0])
    #print(len(embeddings))
    #print(len(embeddings[0]))
    #print(len(embeddings[0][0]))
    df["embeddings"] = embeddings

    #df.to_json(options.save_path, orient='records', lines=True)
            
#parser = ArgumentParser(prog="extract_embeddings.py")
#parser.add_argument('--data',type=str,help="Path to dataset")
#parser.add_argument('--model',type=str,help="Model name")
#parser.add_argument('--save', type=str,help="Path for saving results", default=None)
#parser.add_argument('--task', default="STS", choices=["STS","Summarization","BitextMining","Retrieval"], help='Task (==which query to use)')
#parser.add_argument('--chunk_size', '--chunksize',type=int,help="elements per batch", default = None)
#parser.add_argument('--batch_size', '--batchsize', type=int,help="How many files are handled the same time", default = 2)
#parser.add_argument('--rank',type=int,help="The rank of this job", default = 1)

#options = parser.parse_args()
class Options:
    model="e5"
    save=None
    task="STS"
    chunk_size=None
    batch_size=2

options = Options()

save_file = "testi.pkl" if options.save is None else options.save
assert ".pkl" in save_file, "Include a valid path with .pkl in the end"

if os.path.dirname(save_file):
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

# find model with alias or full name
model = SentenceTransformer(model_name_dict.get(options.model, options.model),trust_remote_code=True)
if options.model in ["qwen","Alibaba-NLP/gte-Qwen2-7B-instruct"]:
    model.max_seq_length = 8192

data = "IN-testset.jsonl"

with open(save_file, "wb") as f:
    transform_with_dict_and_batch(f, model, options)
    #transform_with_dict_and_map(f, model, options)
    #transform_with_pandas(f, model, data, options)