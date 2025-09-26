import torch
import vec2text
import numpy as np
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import json
from jsonargparse import ArgumentParser
from scipy.spatial import distance as eucdistance
from copy import copy
import nltk

filename = "/scratch/project_462000883/amanda/register-data/combined_1000.jsonl"



def divide_sentences(text) -> list:
    return nltk.tokenize.sent_tokenize(text, language='english')


with open(filename, "r") as f:
    lines = f.readlines()
    if "\"text\":" in lines[0] or ".jsonl" in filename:
        ids = [json.loads(l)["id"] for l in lines]
        lines = [json.loads(l)["text"] for l in lines]
    else:
        ids = None
lines = [divide_sentences(l)[0] for l in lines]  # only first sentence

def get_gtr_embeddings(text_list,
                       encoder: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    inputs = tokenizer(text_list,
                       return_tensors="pt",
                       max_length=128,
                       truncation=True,
                       padding="max_length",).to("cuda")
    with torch.no_grad():
        model_output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        hidden_state = model_output.last_hidden_state
        embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])
    return embeddings


encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
embeddings = get_gtr_embeddings(lines, encoder, tokenizer)

with open("seed_embeddings.jsonl", "w") as f:
    if ids:
        for e, l, i in zip(embeddings, lines, ids):
            f.write(json.dumps({"id": i, "text":l, "emb":e.cpu().tolist()})+"\n")
    else:
        for e, l in zip(embeddings, lines):
            f.write(json.dumps({"text":l, "emb":e.cpu().tolist()})+"\n")
