from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import numpy as np
import json
import sys
from tqdm import tqdm
from pathlib import Path
from jsonargparse import ArgumentParser
import os

ap = ArgumentParser(prog="test.py")
ap.add_argument('--model', default="/scratch/project_462000353/amanda/megatron-training/register-training-with-megatron/checkpoints_converted/dtp/iter_0050000")
ap.add_argument('--tokenizer', default="gpt2")
ap.add_argument('--layers', default=-1)
ap.add_argument('--data')
ap.add_argument('--return_text')
ap.add_argument('--seed')
ap.add_argument('--save')


options = ap.parse_args()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
text = "Hello," # I am " # a freelance
model = AutoModel.from_pretrained(options.model)
model2 = AutoModelForCausalLM.from_pretrained(options.model)
tokenizer = AutoTokenizer.from_pretrained(options.tokenizer)
if type(options.layers) == int:
    options.layers = [options.layers]


def tokenize(t):
    if t is None:
        return tokenizer("", return_tensors='pt', truncation=True)
    return tokenizer(t, return_tensors='pt', truncation=True)

def extract(model, text):
    print("in extract ", flush=True)
    with torch.no_grad():
        output = model(**tokenize(text), return_dict_in_generate=True, output_hidden_states=True)
    print(output.keys())
    #hidden_states = output["hidden_states"]
    #print(len(hidden_states))
    #print(hidden_states[-1].shape)
    #print(hidden_states[-1] == sequence)
    #indices = np.array([i for i in options.layers], dtype=int) #np.array([0, len(hidden_states)//2, 3*len(hidden_states)//4, -1], dtype=int)
    #embed = [torch.mean(hidden_states[i],axis=1).cpu().tolist() for i in indices]
    #embed = [torch.mean(hidden_states[i],axis=1).cpu().tolist() for i in indices]
    torch.cuda.empty_cache()
    #return embed, sequence
    return None, output["hidden_states"]

def extract2(model, text):
    print("in extract2 ", flush=True)
    with torch.no_grad():
        output = model.generate(**tokenize(text), return_dict_in_generate=True, output_hidden_states=True, max_new_tokens=50)
    #print(output)
    print(output.keys())
    #sequence = output["sequences"]
    #print(sequence.shape)
    #hidden_states = output["hidden_states"]
    #print(len(hidden_states))
    #print(hidden_states[-1].shape)
    #print(hidden_states[-1] == sequence)
    #indices = np.array([i for i in options.layers], dtype=int) #np.array([0, len(hidden_states)//2, 3*len(hidden_states)//4, -1], dtype=int)
    #embed = [torch.mean(hidden_states[i],axis=1).cpu().tolist() for i in indices]
    #embed = [torch.mean(hidden_states[i],axis=1).cpu().tolist() for i in indices]
    torch.cuda.empty_cache()
    #return embed, sequence
    return output["sequences"], output["hidden_states"]

tokenized_input = tokenize(text).input_ids
print(f'lenght of tokenized input prompt = {len(tokenized_input[0])}, {tokenized_input}')

_, hidden_states = extract(model, text)
generated, hidden_states2 = extract2(model2, text)
print(tokenizer.decode(generated[0]))
print(f'len of generated: {len(generated[0])}')
#print(len(hidden_states), len(hidden_states[0]), len(hidden_states[0][0]),len(hidden_states[0][0][0]))
#print(len(hidden_states2), len(hidden_states2[0]), len(hidden_states2[0][0]), len(hidden_states2[0][0][0]),len(hidden_states2[0][0][0][0]))
#print(len(hidden_states2), len(hidden_states2[1]), len(hidden_states2[1][0]), len(hidden_states2[1][0][0]),len(hidden_states2[1][0][0][0]))

print("For no head:", len(hidden_states), hidden_states[0].shape)
print("For generation:", len(hidden_states2), len(hidden_states2[0]), hidden_states2[0][0].shape)
print("For generation (index 1):", len(hidden_states2), len(hidden_states2[1]), hidden_states2[1][0].shape)
print("For generation (index 2):", len(hidden_states2), len(hidden_states2[2]), hidden_states2[2][0].shape)

#for h1, h2 in zip(hidden_states, hidden_states2[0]):    # these match!!
#    print(h1==h2)
#    print(h1)
#    print(h2)
#    print("----------------------------------------------\n\n")


#tok=tokenizer(text)
#emb = extract(model, tok)
#print(emb)


#output=model.generate(**tokenizer("hello", return_tensors='pt'), return_dict_in_generate=True, output_hidden_states=True)
#print(output.keys())
#print(output["sequences"])
#print(tokenizer.decode(output["sequences"][0]))
#print(output["hidden_states"])
#print(len(output["hidden_states"]))  #20 layers

#model = pipeline('text-generation', options.model)

#output = model("hello", return_dict_in_generate=True)#output_hidden_states=True)
#print(output)
#exit()

