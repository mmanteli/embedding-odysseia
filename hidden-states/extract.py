from transformers import AutoTokenizer, AutoModel
import torch
from jsonargparse import ArgumentParser

ap = ArgumentParser(prog="test.py")
ap.add_argument('--model', default="/scratch/project_462000883/amanda/register-models/dtp/iter_0050000")
ap.add_argument('--tokenizer', default="gpt2")
ap.add_argument('--layers', default=-1)
ap.add_argument('--data')
ap.add_argument('--return_text')
ap.add_argument('--seed')
ap.add_argument('--save')


options = ap.parse_args()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
text ="Hello, I am a freelance"
model = AutoModel.from_pretrained(options.model)
tokenizer = AutoTokenizer.from_pretrained(options.tokenizer)
if type(options.layers) is int:
    options.layers = [options.layers]


def tokenize(t):
    """Tokenize a piece of text."""
    if t is None:
        return tokenizer("", return_tensors='pt', truncation=True)
    return tokenizer(t, return_tensors='pt', truncation=True)

def flatten_by_tokens(state):
    """Flatten a model layer output wrt. tokens. Tokens are in dimension -2."""
    return torch.mean(state, axis=-2)

def extract(model, text, layers=None):
    """Get hidden states of model for given layers."""
    if layers is None:
        layers=[-1]
    with torch.no_grad():
        output = model(**tokenize(text), return_dict_in_generate=True, output_hidden_states=True)
    hidden_states = output["hidden_states"]
    torch.cuda.empty_cache()
    return [flatten_by_tokens(hidden_states[i]) for i in layers]


tokenized_input = tokenize(text).input_ids
print(f'lenght of tokenized input prompt = {len(tokenized_input[0])}, {tokenized_input}')

hidden_states_flattened = extract(model, text, layers=[-1])
# print(len(hidden_states_flattened), hidden_states_flattened[0].shape) # this is correct now







