import gc
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer   # AutoModelForCausalLM
from fastchat.model import get_conversation_template
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM, LlamaTokenizerFast,
                          LlamaForCausalLM)
import os

# Set the random seed for NumPy
np.random.seed(20)
# Set the random seed for PyTorch
torch.manual_seed(20)
# If you are using CUDA (i.e., a GPU), also set the seed for it
torch.cuda.manual_seed_all(20)


access_token = os.environ["HFKEY"]
model_path = 'meta-llama/Llama-2-7b-hf' #'gpt2' #"/scratch/project_462000883/amanda/models/llamackpts/llama2-7b-chat-hf"
num_steps = 50
user_prompt = "Write this sentence twice. "
adv_string_init = "! ! ! ! ! ! ! ! !"
target = "I want strawberry ice cream."  # same number tokens
#template_name = 'llama-2'
device = 'cuda:0'
batch_size = 512
topk = 256
allow_non_ascii = False # you can set this to True to use unicode tokens
test_prefixes = ["target"]



#----------------------------------------unmodified------------------------------------------------#

def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', **kwargs):
    try:
        model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                **kwargs
            ).to(device).eval()
    
        tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            #use_fast=False
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                token=access_token,
                **kwargs,
            ).to(device).eval()
    
        tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            #use_fast=False
        )
        
    # You can define special cases here i.e. mappings for special tokens
    return model, tokenizer

def get_nonascii_toks(tokenizer, device='cpu'):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)
    
    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)
    
    return torch.tensor(ascii_toks, device=device)

def get_embedding_matrix(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte.weight
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in.weight
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_embeddings(model, input_ids):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte(input_ids).half()
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in(input_ids).half()
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:], 
            input_embeds, 
            embeds[:,input_slice.stop:,:]
        ], 
        dim=1)
    
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)
    
    loss.backward()
    
    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    
    return grad

def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0, 
        len(control_toks), 
        len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks


def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None):
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        if filter_cand:
            if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                cands.append(decoded_str)
            else:
                count += 1
        else:
            cands.append(decoded_str)

    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        # print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
    return cands


def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls=None, return_ids=False, batch_size=512):
    
    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")

    if not(test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {control_slice.stop - control_slice.start}), " 
            f"got {test_ids.shape}"
        ))

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        del locs, test_ids ; gc.collect()
        return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
    else:
        del locs, test_ids
        logits = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del ids ; gc.collect()
        return logits

def target_loss(logits, ids, target_slice):
    crit = nn.CrossEntropyLoss(reduction='none')
    loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,target_slice])
    return loss.mean(dim=-1)

def forward(*, model, input_ids, attention_mask, batch_size=512):

    logits = []
    for i in range(0, input_ids.shape[0], batch_size):
        
        batch_input_ids = input_ids[i:i+batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i+batch_size]
        else:
            batch_attention_mask = None

        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)

        gc.collect()

    del batch_input_ids, batch_attention_mask
    
    return torch.cat(logits, dim=0)

def sum_slice(sl, value):
    return slice(sl.start+value, sl.stop+value)

class Manager:
    def __init__(self, tokenizer, instruction, target, adv_string):

        self.tokenizer = tokenizer
        self.conv_template = None
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string
        self._update_slices()

    def _update_slices(self):
        if isinstance(self.tokenizer, LlamaTokenizerFast):
            # here first token is <s>
            self._instruction_slice = slice(1, len(self.tokenizer(self.instruction)["input_ids"]))
            self._adv_slice = sum_slice(slice(1, len(tokenizer(self.adv_string)["input_ids"])), self._instruction_slice.stop)
            self._target_slice = sum_slice(slice(1, len(tokenizer(self.target)["input_ids"])), self._adv_slice.stop)
            self._loss_slice = sum_slice(self._target_slice, -1)  # -1 for causal models
        else:
            self._instruction_slice = slice(0, len(self.tokenizer(self.instruction)["input_ids"]))
            self._adv_slice = sum_slice(slice(0, len(tokenizer(self.adv_string)["input_ids"])), self._instruction_slice.stop)
            self._loss_slice = sum_slice(self._target_slice, -1)  # -1 for causal models

    def get_input_ids(self, adv_string=None):
        if adv_string is not None:
            self.adv_string = adv_string
        self._update_slices()
        prompt = [self.instruction, self.adv_string, self.target]
        toks = self.tokenizer(self.instruction).input_ids + self.tokenizer(self.adv_string).input_ids + self.tokenizer(self.target).input_ids
        input_ids = torch.tensor(toks) #[:self._target_slice.stop])
        return input_ids

#---------------------------------conversationality-----------------------------------------#


def generate(model, tokenizer, input_ids, gen_config=None):
    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 32

    if gen_config.max_new_tokens > 50:
        print('WARNING: max_new_tokens > 32 may cause testing to slow down.')
        
    input_ids = input_ids.to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    output_ids = model.generate(input_ids, 
                                attention_mask=attn_masks, 
                                generation_config=gen_config,
                                pad_token_id=tokenizer.pad_token_id)[0]

    return output_ids

def check_for_attack_success(model, tokenizer, input_ids, test_prefixes, gen_config=None):
    gen_str = tokenizer.decode(generate(model, 
                                        tokenizer, 
                                        input_ids,
                                        gen_config=gen_config)).strip()
    jailbroken = gen_str == test_prefixes[0] #not any([prefix in gen_str for prefix in test_prefixes])
    return jailbroken


    

model, tokenizer = load_model_and_tokenizer(model_path, 
                       low_cpu_mem_usage=True, 
                       use_cache=False,
                       device=device)


suffix_manager = Manager(tokenizer=tokenizer,
              instruction=user_prompt,
              target=target,
              adv_string=adv_string_init)
not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer) 
adv_suffix = adv_string_init


for i in range(num_steps):
    print(f'In step {i}', flush=True)
    
    print(f'Getting input ids')
    # Step 1. Encode user prompt (behavior + adv suffix) as tokens and return token ids.
    input_ids = suffix_manager.get_input_ids()
    input_ids = input_ids.to(device)
    print(f'Decoded input is: {suffix_manager.tokenizer.decode(input_ids)}')
    print(f'Adversarial slice is: {suffix_manager.tokenizer.decode(input_ids[suffix_manager._adv_slice])} ({suffix_manager._adv_slice})')
    print(f'Target slice is: {suffix_manager.tokenizer.decode(input_ids[suffix_manager._target_slice])}')
    print(f'Loss slice is: {suffix_manager.tokenizer.decode(input_ids[suffix_manager._loss_slice])}')
    
    # Step 2. Compute Coordinate Gradient
    #print(f'gradient calculation:\n\ntarget slice {suffix_manager._target_slice}\n\ncontrol slice {suffix_manager._adv_slice}\n\nloss slice {suffix_manager._loss_slice}\n')
    coordinate_grad = token_gradients(model, 
                    input_ids, 
                    suffix_manager._adv_slice, 
                    suffix_manager._target_slice, 
                    suffix_manager._loss_slice)
    
    # Step 3. Sample a batch of new tokens based on the coordinate gradient.
    # Notice that we only need the one that minimizes the loss.
    with torch.no_grad():
        
        # Step 3.1 Slice the input to locate the adversarial suffix.
        adv_suffix_tokens = input_ids[suffix_manager._adv_slice].to(device)
        
        # Step 3.2 Randomly sample a batch of replacements.
        new_adv_suffix_toks = sample_control(adv_suffix_tokens, 
                       coordinate_grad, 
                       batch_size, 
                       topk=topk, 
                       temp=1, 
                       not_allowed_tokens=not_allowed_tokens)
        
        
        # Step 3.3 This step ensures all adversarial candidates have the same number of tokens. 
        # This step is necessary because tokenizers are not invertible
        # so Encode(Decode(tokens)) may produce a different tokenization.
        # We ensure the number of token remains to prevent the memory keeps growing and run into OOM.
        new_adv_suffix = get_filtered_cands(tokenizer, 
                                            new_adv_suffix_toks, 
                                            filter_cand=True, 
                                            curr_control=adv_suffix)
        
        # Step 3.4 Compute loss on these candidates and take the argmin.
        logits, ids = get_logits(model=model, 
                                 tokenizer=tokenizer,
                                 input_ids=input_ids,
                                 control_slice=suffix_manager._adv_slice, 
                                 test_controls=new_adv_suffix, 
                                 return_ids=True,
                                 batch_size=512) # decrease this number if you run into OOM.

        losses = target_loss(logits, ids, suffix_manager._target_slice)

        best_new_adv_suffix_id = losses.argmin()
        best_new_adv_suffix = new_adv_suffix[best_new_adv_suffix_id]

        current_loss = losses[best_new_adv_suffix_id]

        # Update the running adv_suffix with the best candidate
        adv_suffix = best_new_adv_suffix
        is_success = check_for_attack_success(model, 
                                 tokenizer,
                                 suffix_manager.get_input_ids(adv_string=adv_suffix).to(device),
                                 test_prefixes)
        
    if is_success:
        break

input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix).to(device)

gen_config = model.generation_config
gen_config.max_new_tokens = 256

completion = tokenizer.decode((generate(model, 
                                        tokenizer, 
                                        input_ids,
                                        gen_config=gen_config))).strip()

print(f"\nCompletion: {completion}")   