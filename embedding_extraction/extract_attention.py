import torch
from jsonargparse import ArgumentParser
from  transformers import AutoTokenizer
import sys
import json
import traceback
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
import nltk
#nltk.download('punkt')
#nltk.download('punkt_tab')
import torch.nn.functional as F

try:
    from model_utils import model_name_dict, get_all_prompts
    from extract import pickle_dump_with_segmented_id, text2chunks, text2wordchunks, text2tokenchunks, text2sentences
except ImportError:
    from embedding_extraction.model_utils import model_name_dict, get_all_prompts

from datasets.utils.logging import disable_progress_bar      #prevent the libraries from displaying a progress bar
disable_progress_bar()

from datasets.utils.logging import set_verbosity_error
set_verbosity_error()                                        #prevent the libraries' messages from being displayed

# Help
# https://sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html



#------------------------------------------ Main loop ------------------------------------------ #


def average_pool(token_embeddings, attention_mask):
    """
    Pool token embeddings to sentence level embeddings with average pool.
    Rewrite from SentenceTransformers.models.Pooling.py.
    """
    input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
                )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    return sum_embeddings / sum_mask

def normalize(features):
    """Normalize to unit length."""
    return F.normalize(features, p=2, dim=1)

def transform(f, options, f_train=False):
    """
    Read input from sys.stdin and calculate embeddings for them.
    batch_size controls how many documents (or document segments if chunk_size is set) are handled
    at the same time.
    Dumped to a pickle document id-wise.
    """
    # find model with alias or full name
    model = SentenceTransformer(
                                model_name_dict.get(options.model, options.model),
                                prompts=get_all_prompts(),
                                default_prompt_name=options.task,
                                trust_remote_code=True,
                                device = "cuda:0" if torch.cuda.is_available() else "cpu",
                                )
    if options.model in ["qwen","Alibaba-NLP/gte-Qwen2-7B-instruct"]:
        model.max_seq_length = 8192
    tokenizer = model.tokenizer

   # Get the base transformer; this is where we can access attention
    transformer = model._first_module()
    hf_model = transformer.auto_model
    # set to eval mode
    hf_model.eval()

    # Define attention collection; for each forward pass we also save the attention
    attention_outputs = []
    def hook_fn(module, input, output):
        attention_outputs.append(output[0])
    hook = hf_model.encoder.layer[-1].attention.register_forward_hook(hook_fn)

    texts = []
    ids = []
    labels = []
    offsets = []
    chunk_fn = {"tokens": "", # these defined below
                "truncate": "",
                "words": text2wordchunks,
                "sentences": text2sentences,
                "chars": text2chunks}[options.split_by]
    for idx, line in enumerate(sys.stdin):
        try:
            j = json.loads(line)
            if options.split_byin ["sentences", "chars", "words"]:
                assert options.chunk_size, "Give --chunk_size with --split_by != truncate."
                chunk, id_, label, offset  = chunk_fn(j, options.chunk_size, options.overlap)
                texts.extend(chunk)
                ids.extend(id_)
                labels.extend(label)
                offsets.extend(offset)
            elif options.split_by == "tokens":
                max_length = options.chunk_size if options.chunk_size else tokenizer.model_max_length - 2
                # -2: room for special tokens
                overlap = options.overlap if options.overlap else int(max_length/2)-1
                chunk, id_, label, offset  = text2tokenchunks(j, tokenizer, max_length, overlap)
                texts.extend(chunk)
                ids.extend(id_)
                labels.extend(label)
                offsets.extend(offset)
            else:
                j["offset"] = None
                texts.append(j["text"])
                ids.append(j["id"])
                labels.append(j["register"])
                offsets.append(j["offset"])
        except:
            print(f'Problem with text on idx {idx}\n', flush=True)
            traceback.print_exc()

        if len(texts) >= options.batch_size:
            #if options.debug: print(f"Doing a batch at index {idx}", flush=True)
            tokenized = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(hf_model.device)
            with torch.no_grad():
                output = hf_model(**tokenized)
                token_embeddings = output.last_hidden_state
                attention_mask = output['attention_mask']
                

            pickle_dump_with_segmented_id(f, ids, offsets, labels, texts,
                                          attention_outputs, f_train=f_train, th=options.threshold)

            # re-init
            texts = []
            ids = []
            labels = []
            offsets = []
            attention_outputs = []

    if len(ids) > 0:   # we have leftovers; e.g. last chunk was not over batch size
        if options.debug:
            print("Dumping leftovers", flush=True)
        tokenized = tokenizer(texts)
        _ = hf_model(**tokenized)
        pickle_dump_with_segmented_id(f, ids, offsets, labels, texts,
                                          attention_outputs, f_train=f_train, th=options.threshold)
    hook.remove()

#-------------------------------------------- Start -------------------------------------------- #

if __name__=="__main__":

    parser = ArgumentParser(prog="extract.py")
    parser.add_argument('--model',type=str,help="Model name")
    parser.add_argument('--data', type=str,help="Path for saving results", default=None)
    parser.add_argument('--task', default="STS", choices=["STS","Summarization","BitextMining","Retrieval"],
                        help='Task (==which query to use)')
    parser.add_argument('--batch_size', '--batchsize', type=int,
                        help="How many files are handled the same time", default = 500)
    parser.add_argument('--model_batch_size', type=int, default=32)  # tested to be the fastest out of 32 64 128
    parser.add_argument('--split_by', default="sentences",
                        choices=["tokens", "sentences", "chars", "words", "truncate"],
                        help='What to use for splitting too long texts, truncate=nothing')
    parser.add_argument('--chunk_size', type=int, help="How many units (tokens, words, chars) per batch. For sentences,\
                        splits up by char, for tokens, automatically set to model_max_len -2", default = None)
    parser.add_argument('--overlap', '--context_overlap', type=int, help="How much overlap per segment \
                        (None = model_max_len/2)", default = None)
    parser.add_argument('--temporary_training_set', type=str, default=None,
                        help='Sample training data to temporary .pkl')
    parser.add_argument('--threshold', type=float, default=0.1, help='Sample fraction for temp training data')
    parser.add_argument('--debug', type=bool, default=False, help="Verbosity etc.")

    options = parser.parse_args()
    print(options, flush=True)

    save_file = "testi.pkl" if options.data is None else options.data
    temp_save_file = False if options.temporary_training_set is None else options.temporary_training_set
    assert ".pkl" in save_file, "Include a valid path with .pkl in the end"

    if os.path.dirname(save_file):
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
    if os.path.dirname(temp_save_file):
        os.makedirs(os.path.dirname(temp_save_file), exist_ok=True)

    if not temp_save_file:
        with open(save_file, "wb") as f:
            transform(f, options)
    else:
        with open(save_file, "wb") as f, open(temp_save_file, "wb") as f_train:
            transform(f, options, f_train=f_train)
