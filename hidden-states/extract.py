from transformers import AutoTokenizer, AutoModel
import torch
import json
import pandas as pd
from jsonargparse import ArgumentParser
import datasets
from torch.utils.data import DataLoader

ap = ArgumentParser(prog="test.py")
ap.add_argument('--model', default="/scratch/project_462000883/amanda/register-models/OP/iter_0050000")
ap.add_argument('--tokenizer', default="gpt2")
ap.add_argument('--layers', default=-1)
ap.add_argument('--data')
ap.add_argument('--return_text')
ap.add_argument('--seed')
ap.add_argument('--save')


options = ap.parse_args()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'Using {device}')
# load model and tokenizer
model = AutoModel.from_pretrained(options.model)
tokenizer = AutoTokenizer.from_pretrained(options.tokenizer)
if type(options.layers) is int:
    options.layers = [options.layers]
print("Model and tokenizer loaded.")
# load data
#ds = datasets.load_dataset('json', data_files=options.data)
#print(ds)
#ds = ds["train"].select(range(10))
#print("dataset loaded")
#print(ds)

# load data
data = []
with open(options.data) as f:
    for line in f:
        data.append(json.loads(line))

data = data[:10]
# functions for embedding :)
def tokenize(t):
    """Tokenize a piece of text."""
    if t is None:
        return tokenizer("", return_tensors='pt', truncation=True)
    return tokenizer(t, return_tensors='pt', truncation=True)


# Does not work because padding token does not exist!
def tokenize_fn(t):
    """Tokenize a dict with field 'text'."""
    if t is None:
        return tokenizer("", return_tensors='pt', truncation=True, padding=True)
    return tokenizer(t["text"], return_tensors='pt', truncation=True, padding=True)

def flatten_by_tokens(state):
    """Flatten a model layer output wrt. tokens. Tokens are in dimension -2."""
    return torch.mean(state, axis=-2).numpy().tolist()

def extract(model, tokenized, layers):
    """Get hidden states of model for given layers."""
    tokenized.to(model.device)
    with torch.no_grad():
        output = model(**tokenized, return_dict_in_generate=True, output_hidden_states=True)
    hidden_states = output["hidden_states"]
    del output
    return_value = [flatten_by_tokens(hidden_states[i]) for i in layers]
    del hidden_states
    torch.cuda.empty_cache()
    return return_value



#ds = ds.map(tokenize_fn, batched=True)
#print("tokenized")
#ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
#loader = DataLoader(ds, batch_size=4)
#for batch in loader:
#    print("new_batch")
#    batch = {k: v.to(model.device) for k, v in batch.items()}
#    output = extract(model, batch, layers=options.layers)


results = []
for t in data:
    print("new iteration")
    tokenized = tokenize(t["text"])
    print("\tNow embedding")
    output = extract(model, tokenized, options.layers)
    print("\tEmbedded, saving...")
    assert len(options.layers) == len(output)
    embed_results = {}
    for o,l in zip(output, options.layers):
        embed_results[f"layer_{l}"] = o
    results.append({**t, **embed_results})

print("embedding done")
df = pd.DataFrame.from_dict(results)
print(df)
df.to_csv(options.save)
#ds.save_to_disk(options.save)





