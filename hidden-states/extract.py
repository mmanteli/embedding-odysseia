from transformers import AutoTokenizer, AutoModel
import torch
from jsonargparse import ArgumentParser

ap = ArgumentParser(prog="test.py")
ap.add_argument('--model',
    default="/scratch/project_462000353/amanda/megatron-training/register-training-with-megatron/checkpoints_converted/dtp/iter_0050000")
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

def extract(model, text, layers=-1):
    """Get hidden states of model for given layers."""
    if type(layers) is not list:
        layers=[layers]
    print("in extract ", flush=True)
    with torch.no_grad():
        output = model(**tokenize(text), return_dict_in_generate=True, output_hidden_states=True)
    hidden_states = output["hidden_states"]
    #print(len(hidden_states))
    #print(hidden_states[-1].shape)
    #print(hidden_states[-1] == sequence)
    #indices = np.array([i for i in options.layers], dtype=int)
    # #np.array([0, len(hidden_states)//2, 3*len(hidden_states)//4, -1], dtype=int)
    #embed = [torch.mean(hidden_states[i],axis=1).cpu().tolist() for i in indices]
    #embed = [torch.mean(hidden_states[i],axis=1).cpu().tolist() for i in indices]
    torch.cuda.empty_cache()
    #return embed, sequence
    return [hidden_states[i] for i in layers]


tokenized_input = tokenize(text).input_ids
print(f'lenght of tokenized input prompt = {len(tokenized_input[0])}, {tokenized_input}')

hidden_states = extract(model, text)
print(len(hidden_states), hidden_states[0].shape)





