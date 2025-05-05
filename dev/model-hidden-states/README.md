# I want to get embeddings from a register trained model

From Sampo

```
OK! First, start an interactive session on a GPU node, e.g.
srun \
    --account=project_462000883 \
    --partition=dev-g \
    --ntasks=1 \
    --gres=gpu:mi250:1 \
    --time=1:00:00 \
    --mem=32G \
    --pty \
    bash
then
module use /appl/local/csc/modulefiles; module load pytorch
python3
and (using gpt2 as example)
>>> from transformers import pipeline
>>> p = pipeline('text-generation', 'gpt2')
>>> p('hello')
just replace gpt2 with the directory with any HF model there :+1:


srun \
    --account=project_462000883 \
    --partition=dev-g \
    --ntasks=1 \
    --gres=gpu:mi250:1 \
    --time=0:10:00 \
    --mem=32G \
    --pty \
    bash
module use /appl/local/csc/modulefiles; module load pytorch
python 
from transformers import pipeline
p = pipeline('text-generation', '/scratch/project_462000353/amanda/megatron-training/register-training-with-megatron/checkpoints_converted/dtp/iter_0050000')
p('hello')
```

## What do the dimensions mean?

I tried both with ``AutoModel`` (no model head) and ``AutoModelForCausalLM`` (generation head). Pipeline did not work with ``return_dict_in_generate=True, output_hidden_states=True``, so I did not use it.
Using ``dtp`` step ``0050000`` here.

```
prompt = "Hello,"
lenght of tokenized input prompt: 2
# generated:
generated output: Hello, I am a freelance writer and editor with a passion for travel, food, and culture. I have written for a variety of publications, including The New York Times, The Atlantic, and The New York Observer. I have also written for a number of
length of generated: 52
# -> generation adds 50 tokens, which is what I set it to do

# output dimensions

For no head: 25 x torch.Size([1, 2, 2048])    # here 25 is the number of tuples (layers), 2 is the dimensions of input (2 tokens), 2048 is the model dim.
For generation: 50 x 25 x torch.Size([1, 2, 2048])   # here 50 is the number of generated, 25 is the same, but why is there 2 here as well?
For generation (index 1): 50 25 torch.Size([1, 1, 2048])
For generation (index 2): 50 25 torch.Size([1, 1, 2048])
```

#### Question: what is up with the dimension of generation?

- 1 is the batch size.
- ChatGPT tells me that in generation, the model only returns the hidden states for the initial input, NOT what it has generated before. I.e. 
```
In theory
1st generation: hidden states for 2 tokens (==input), output 3rd token
2nd generation: hidden states for 3 tokens (==input + previous), output 4th token
3rd generation: hidden states for 4 tokens (=input + 2 previous), output 5th token
etc.

However, what apparently happens is it only returns first 2 every time.
```