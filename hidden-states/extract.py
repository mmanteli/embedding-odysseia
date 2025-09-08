from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import json
import ast
from jsonargparse import ArgumentParser
#from torch.utils.data import DataLoader
import pickle

model_dict = lambda x: {"50k": f"/scratch/project_462000883/amanda/register-training-with-megatron/checkpoints_converted/{x}/iter_0050000",
                        "51k": f"/scratch/project_462000883/amanda/register-training-with-megatron/checkpoints_converted/{x}/iter_0051000",
                        50000: f"/scratch/project_462000883/amanda/register-training-with-megatron/checkpoints_converted/{x}/iter_0050000",
                        51000: f"/scratch/project_462000883/amanda/register-training-with-megatron/checkpoints_converted/{x}/iter_0051000",
                        }


ap = ArgumentParser(prog="extract.py")
ap.add_argument('--model')
ap.add_argument('--register')
ap.add_argument('--iter')
ap.add_argument('--tokenizer', default="gpt2")
ap.add_argument('--layers', default=-1)
ap.add_argument('--data_path')
ap.add_argument('--data_is_tokenized', action='store_true')
ap.add_argument('--return_text', action='store_true')
ap.add_argument('--seed')
ap.add_argument('--sample', type=int, help="select a sample from data")
ap.add_argument('--save')


options = ap.parse_args()
print(options, flush=True)
if options.model is None:
    assert options.register is not None, "Give --register and --iter if no --model is specified."
    assert options.iter is not None, "Give --register and --iter if no --model is specified."
    options.model = model_dict(options.register)[options.iter]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'Using {device}', flush=True)

# load model and tokenizer
model = AutoModel.from_pretrained(options.model)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(options.tokenizer)
if type(options.layers) is int:
    options.layers = [options.layers]
print("Model and tokenizer loaded.", flush=True)


# load data
data = []
with open(options.data_path) as f:
    print("reading a data", flush=True)
    for line in f:
        data.append(ast.literal_eval(line))
        if options.sample:
            if len(data)>= options.sample:
                break
#if options.sample:
#    data = data[:options.sample] # TODO, this is for testing

print("data is read", flush=True)
print(len(data), flush=True)

# functions for embedding :)
def tokenize(t):
    """Tokenize a piece of text."""
    if t is None:
        return tokenizer("", return_tensors='pt', truncation=True)
    return tokenizer(t, return_tensors='pt', truncation=True)

def flatten_by_tokens(state):
    """Flatten a model layer output wrt. tokens. Tokens are in dimension -2."""
    return torch.mean(state, axis=-2).numpy().tolist()

def extract(model, tokenized, layers):
    """Get hidden states of model for given layers."""
    try:
        tokenized.to(model.device)
    except:
        tokenized = {"input_ids":torch.tensor([tokenized])}
        tokenized.to(model.device)
    with torch.no_grad():
        output = model(**tokenized, return_dict_in_generate=True, output_hidden_states=True)
    hidden_states = output["hidden_states"]
    del output
    return_value = [flatten_by_tokens(hidden_states[i].cpu()) for i in layers]
    del hidden_states
    torch.cuda.empty_cache()
    return return_value


#results = []
with open(options.save, 'wb') as f:
    for t in data:
        print("new iteration", flush=True)
        tokenized = tokenize(t["text"]) if not options.data_is_tokenized else t["text"]
        print("\tNow embedding, flush=True")
        print(tokenized, flush=True)
        output = extract(model, tokenized, options.layers)
        print("\tEmbedded, saving...", flush=True)
        assert len(options.layers) == len(output)
        embed_results = {}
        for o,l in zip(output, options.layers):
            embed_results[f"layer_{l}"] = o
        if options.return_text:
            t["detok"] = tokenizer.decode(tokenized)
        #results.append({**t, **embed_results})
        #print(json.dumps({**t, **embed_results}), file=f)
        pickle.dump(str(json.dumps({**t, **embed_results})), f)

