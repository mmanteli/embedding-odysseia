import vec2text
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import json
from pathlib import Path
from typing import Iterator, Dict, List, Union
from jsonargparse import ArgumentParser
import nltk

parser = ArgumentParser()
parser.add_argument("--files")
parser.add_argument("--search_iter", default=20, type=int)
parser.add_argument("--length", default=1, type=int)  # num sentences
options = parser.parse_args()

def divide_sentences(text) -> list:
    sentences = nltk.tokenize.sent_tokenize(text, language='english')
    segmented = []
    for s in sentences:
        segmented.append(s)
    return segmented


def jsonl_batch_reader(
    files: Union[str, Path, List[Union[str, Path]]],
    batch_size: int,
    leng: int
) -> Iterator[List[Dict]]:

    if isinstance(files, (str, Path)):
        files = [files]

    buffer = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # skip empty lines
                    if leng > 0:
                        buffer.extend(divide_sentences(json.loads(line)["text"])[:leng])
                    else:
                        buffer.append(json.loads(line)["text"])
                    if len(buffer) >= batch_size:
                        yield buffer
                        buffer = []
    if buffer:  # yield remaining records
        yield buffer


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
corrector = vec2text.load_pretrained_corrector("gtr-base")


"""
embeddings = get_gtr_embeddings([
       "Jack Morris is a PhD student at Cornell Tech in New York City",
       "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity"
], encoder, tokenizer)

vec2text.invert_embeddings(
    embeddings=embeddings.cuda(),
    corrector=corrector,
    num_steps=20,
)
"""


print(options, flush=True)
average_similarity_cosine = []
average_similarity_euclidean = []

for batch in jsonl_batch_reader(options.files, batch_size=100, leng=options.length):
    embeddings = get_gtr_embeddings(batch, encoder, tokenizer)
    inverted = vec2text.invert_embeddings(
                    embeddings=embeddings.cuda(),
                    corrector=corrector,
                    num_steps=options.search_iter,
                    )
    reembeddings = get_gtr_embeddings(inverted, encoder, tokenizer)
    idx = 0
    for b, i in zip(batch, inverted):
        print(f'Original:\n{b}')
        print(f'Inverted:\n{i}')
        e = embeddings[idx, :]
        r = reembeddings[idx,:]
        cs = (torch.sum(e*r)/(torch.linalg.norm(e)*torch.linalg.norm(r))).detach().cpu().tolist()
        es = (torch.linalg.norm(e-r)).detach().cpu().tolist()
        print(f'\nCosine similarity: {cs}')
        print(f'Euclidean distance: {es}')
        average_similarity_cosine.append(cs)
        average_similarity_euclidean.append(es)
        idx+=1
        print("\n------------------------------------\n")

    if len(average_similarity_cosine) > 10000:
        break



print(f'cosine: {np.mean(average_similarity_cosine)}, ({np.std(average_similarity_cosine)})')
print(f'euclidean: {np.mean(average_similarity_euclidean)}, ({np.std(average_similarity_euclidean)})')

