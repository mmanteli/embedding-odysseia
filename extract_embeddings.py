from sentence_transformers import SentenceTransformer
from jsonargparse import ArgumentParser
import torch
import json
import pandas as pd
import sys 

# Define possible models here
model_name_dict = {"e5": "intfloat/multilingual-e5-large-instruct",
                   "qwen" : "Alibaba-NLP/gte-Qwen2-7B-instruct",
                   "jina" : "jinaai/jina-embeddings-v3"
}

ap = ArgumentParser(prog="extract_embeddings.py")
ap.add_argument('--data', type=str, required=True, metavar='DIR',
                help='')
ap.add_argument('--model', choices=model_name_dict.keys(),
                help='')
ap.add_argument('--task', default="STS", choices=["STS","Summarization","BitextMining","Retrieval"],
                help='')
ap.add_argument('--save_path', required=True, 
                help='where to save the results in jsonl-format')


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



def extract(options):

    df = pd.read_json(options.data, lines=True)
    df = df.head(10)
    print(df.columns)

    model = SentenceTransformer(model_name_dict[options.model],trust_remote_code=True)
    if options.model == "qwen":
        model.max_seq_length = 8192

    embeddings = []
    for index, row in df.iterrows():
        query = get_query(options.task, row["text"])
        input_texts = query + [] # documents
        embeddings.append(model.encode(input_texts, convert_to_tensor=False, normalize_embeddings=False)[0])

    print(len(embeddings))
    print(len(embeddings[0]))
    #print(len(embeddings[0][0]))
    df["embeddings"] = embeddings

    df.to_json(options.save_path, orient='records', lines=True)
    


if __name__=="__main__":
    options = ap.parse_args(sys.argv[1:])
    extract(options)
