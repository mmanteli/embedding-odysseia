import torch
from jsonargparse import ArgumentParser
from  transformers import AutoTokenizer
import sys
import json
import traceback
import numpy as np
import pickle
import datasets
import os
from sentence_transformers import SentenceTransformer
import nltk
#nltk.download('punkt')
#nltk.download('punkt_tab')
try:
    from model_utils import model_name_dict, get_all_prompts
except ImportError:
    from embedding_extraction.model_utils import model_name_dict, get_all_prompts

from datasets.utils.logging import disable_progress_bar      #prevent the libraries from displaying a progress bar
disable_progress_bar()

from datasets.utils.logging import set_verbosity_error
set_verbosity_error()                                        #prevent the libraries' messages from being displayed

# Help
# https://sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html


#--------------------------------- Chuncking of long documents --------------------------------- #

def text2dataset(line, chunk_size):
    """
    Turn a text into a dataset of text chunks.
    """
    txt = line["text"]
    chunks=[]
    offsets=[]
    for chunk_offset in range(0,len(txt),chunk_size):
        chunks.append(txt[chunk_offset:chunk_offset+chunk_size])
        offsets.append(chunk_offset)
    return datasets.Dataset.from_dict({"text":chunks,
                                       "id":[line["id"]]*len(chunks),
                                       "register": [line["register"]]*len(chunks),
                                       "offset": offsets})

def text2chunks_old(line, chunk_size):
    """
    Turn a text into chunked segments.
    """
    txt = line["text"]
    chunks=[]
    offsets=[]
    for chunk_offset in range(0,len(txt),chunk_size):
        chunks.append(txt[chunk_offset:chunk_offset+chunk_size])
        offsets.append(chunk_offset)
    return chunks, [line["id"]]*len(chunks), [line["register"]]*len(chunks), offsets

def text2sentences(line, chunk_size):
    """
    Turn a text into segments.
    """

    def divide_sentences(text, chunk_size) -> list:
        sentences = nltk.tokenize.sent_tokenize(text, language='english')
        segmented = []
        for s in sentences:
            if len(s) < chunk_size:
                segmented.append(s)
            else:
                for chunk_offset in range(0,len(s),chunk_size):
                    segmented.append(s[chunk_offset:chunk_offset+chunk_size])
        return segmented

    txt = line["text"]
    chunks=[]
    offsets=[]
    current_offset=0
    for s in divide_sentences(txt, chunk_size):
        chunks.append(s)
        offsets.append(current_offset)
        current_offset+=len(s)
    return chunks, [line["id"]]*len(chunks), [line["register"]]*len(chunks), offsets

def text2chunks(line, chunk_size, overlap):
    """
    Turn text to possibly overlapping chunks.
    """
    txt = line["text"]
    max_length=chunk_size
    if overlap:
        # initialize
        indices_upper_limits=[max_length]
        indices_lower_limits=[0]
        k = 1 # multiplyer
        # this could be made with for loop, if we knew the max value of k
        # => possible to calculate but I do not trust myself like that
        while indices_upper_limits[-1] < len(txt):
            indices_upper_limits.append((k+1)*max_length - k*overlap)
            indices_lower_limits.append(indices_upper_limits[-1]-max_length)
            k+=1
    else:   # either overlap=0 or None
        indices_upper_limits = [i for i in range(max_length, len(txt)+max_length, max_length)]
        indices_lower_limits = [i for i in range(0, len(txt), max_length)]

    chunks=[]
    offsets=[]
    current_offset = 0
    for start, end in zip(indices_lower_limits, indices_upper_limits):
        chunked_text = txt[start:end]
        chunks.append(chunked_text)
        offsets.append(current_offset)
        current_offset += len(chunked_text)
    return chunks, [line["id"]]*len(chunks), [line["register"]]*len(chunks), offsets

def text2wordchunks(line, max_length, overlap):
    """
    Turn a text into chunked segments wrt whitespace, with defined overlap.
    """
    txt = line["text"]

    input_ids = txt.split(" ")

    if overlap:
        # initialize
        indices_upper_limits=[max_length]
        indices_lower_limits=[0]
        k = 1 # multiplyer
        # this could be made with for loop, if we knew the max value of k
        # => possible to calculate but I do not trust myself like that
        while indices_upper_limits[-1] < len(input_ids):
            indices_upper_limits.append((k+1)*max_length - k*overlap)
            indices_lower_limits.append(indices_upper_limits[-1]-max_length)
            k+=1
    else:   # either overlap=0 or None
        indices_upper_limits = [i for i in range(max_length, len(input_ids)+max_length, max_length)]
        indices_lower_limits = [i for i in range(0, len(input_ids), max_length)]

    # go over the indices and collect results.
    chunks=[]
    offsets=[]
    current_offset = 0
    for start, end in zip(indices_lower_limits, indices_upper_limits):
        chunked_tokens = input_ids[start:end]
        chunked_text = " ".join(chunked_tokens)
        chunks.append(chunked_text)
        offsets.append(current_offset)
        current_offset += len(chunked_text)
    return chunks, [line["id"]]*len(chunks), [line["register"]]*len(chunks), offsets


def text2tokenchunks(line, tokenizer, max_length, overlap):
    """
    Turn a text into chunked segments wrt token count, with defined overlap.
    """
    txt = line["text"]
    tokenized = tokenizer(txt, return_overflowing_tokens=True)

    input_ids = tokenized["input_ids"][0]  # remove nesting

    if overlap:
        # initialize
        indices_upper_limits=[max_length]
        indices_lower_limits=[0]
        k = 1 # multiplyer
        # this could be made with for loop, if we knew the max value of k
        # => possible to calculate but I do not trust myself like that
        while indices_upper_limits[-1] < len(input_ids):
            indices_upper_limits.append((k+1)*max_length - k*overlap)
            indices_lower_limits.append(indices_upper_limits[-1]-max_length)
            k+=1
    else:   # either overlap=0 or None
        indices_upper_limits = [i for i in range(max_length, len(input_ids)+max_length, max_length)]
        indices_lower_limits = [i for i in range(0, len(input_ids), max_length)]

    # go over the indices and collect results.
    chunks=[]
    offsets=[]
    current_offset = 0
    for start, end in zip(indices_lower_limits, indices_upper_limits):
        chunked_tokens = input_ids[start:end]
        chunked_text = tokenizer.decode(chunked_tokens, skip_special_tokens = True, clean_up_tokenisation_spaces=True)
        chunks.append(chunked_text)
        offsets.append(current_offset)
        current_offset += len(chunked_text)
    return chunks, [line["id"]]*len(chunks), [line["register"]]*len(chunks), offsets


#------------------------------------- Pickle dump results ------------------------------------- #

def pickle_dump_wrt_id(f, ids, offsets, labels, texts, embeddings):
    """
    Pickle datasets wrt the id.
    When calling pickle.load(f), one whole document is returned
    (as opposed to one segment of a document or one batch of documents).
    """
    data_to_dump = []
    previous_id = None

    for i, o, r, t, e in zip(ids, offsets, labels, texts, embeddings):
        if previous_id is not None and i != previous_id:  # if ID changes, dump the current data
            pickle.dump(data_to_dump, f)
            data_to_dump = []  # reset for new ID

        data_to_dump.append({"id": i, "offset": o, "register": r, "text": t, "embeddings": e})
        previous_id = i

    # dump the leftovers (should be the last id)
    if data_to_dump:
        pickle.dump(data_to_dump, f)

def pickle_dump_with_segmented_id(f, ids, offsets, labels, texts, embeddings, f_train=False, th=0.1):
    """
    Pickle datasets with a segmented id: id = id + offset.
    Calling pickle.load(f) returns one sentence, but others are easy to find
    when moved the data to an sqlitedict later.
    """
    if f_train:
        for i, o, r, t, e in zip(ids, offsets, labels, texts, embeddings):
            data_to_dump = {"id": str(i)+"-"+str(o), "offset": o, "register": r, "text": t, "embeddings": e}
            pickle.dump(data_to_dump, f)
            if np.random.random() < th:
                pickle.dump(data_to_dump, f_train)
    else:
        for i, o, r, t, e in zip(ids, offsets, labels, texts, embeddings):
            data_to_dump = {"id": str(i)+"-"+str(o), "offset": o, "register": r, "text": t, "embeddings": e}
            pickle.dump(data_to_dump, f)


#------------------------------------ Embedding calculation ------------------------------------ #

def embed(model, input_texts, options):
    """Embed given texts with given model."""
    #input_texts = [get_query(options.task, t) for t in texts]
    #embedded_texts = model.encode(input_texts, convert_to_tensor=False, normalize_embeddings=False)
    #input_texts = texts
    return model.encode(input_texts,
                        convert_to_tensor=False,
                        normalize_embeddings=False,
                        batch_size=options.model_batch_size)


#------------------------------------------ Main loop ------------------------------------------ #

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
    if options.split_by == "tokens":
        tokenizer = AutoTokenizer.from_pretrained(model_name_dict.get(options.model, options.model))

    texts = []
    ids = []
    labels = []
    offsets = []
    for idx, line in enumerate(sys.stdin):
        #if options.debug: print(f"In document {idx}", flush=True)
        try:
            j = json.loads(line)
            if options.split_by == "sentences":
                assert options.chunk_size, "Give --chunk_size with --split_by=sentences"
                chunk, id_, label, offset  = text2sentences(j, options.chunk_size)
                texts.extend(chunk)
                ids.extend(id_)
                labels.extend(label)
                offsets.extend(offset)
            elif options.split_by == "chars":
                assert options.chunk_size, "Give --chunk_size with --split_by=chars"
                chunk, id_, label, offset  = text2chunks(j, options.chunk_size, options.overlap)
                texts.extend(chunk)
                ids.extend(id_)
                labels.extend(label)
                offsets.extend(offset)
            elif options.split_by == "tokens":
                max_length = options.chunk_size if options.chunk_size else tokenizer.model_max_length - 2 # room for special tokens
                overlap = options.overlap if options.overlap else int(max_length/2)-1
                chunk, id_, label, offset  = text2tokenchunks(j, tokenizer, max_length, overlap)
                texts.extend(chunk)
                ids.extend(id_)
                labels.extend(label)
                offsets.extend(offset)
            elif options.split_by == "words":
                assert options.chunk_size, "Give --chunk_size with --split_by=words"
                chunk, id_, label, offset  = text2wordchunks(j, options.chunk_size, options.overlap)
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
            embedded_texts = embed(model, texts, options)
            pickle_dump_with_segmented_id(f, ids, offsets, labels, texts,
                                          embedded_texts, f_train=f_train, th=options.threshold)

            # re-init
            texts = []
            ids = []
            labels = []
            offsets = []

    if len(ids) > 0:   # we have leftovers; e.g. last chunk was not over batch size
        if options.debug:
            print("Dumping leftovers", flush=True)
        embedded_texts = embed(model, texts, options)
        pickle_dump_with_segmented_id(f, ids, offsets, labels, texts,
                                      embedded_texts, f_train=f_train, th=options.threshold)


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
