import vec2text
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import json
from pathlib import Path
from typing import Iterator, Dict, List, Union
from jsonargparse import ArgumentParser
from scipy.spatial import distance as eucdistance
from copy import copy
import os

parser = ArgumentParser()
parser.add_argument('--sentence', default=None, help="sentence to be used as an origin")
parser.add_argument('--embedding_file', default=None, help="path to file containing embeddings")
parser.add_argument('--file_index', default=None, type=int, help="Identifier for data in the embedding file.")
parser.add_argument("--search_iter", default=30, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--n_lines", default=0, type=int)  # 0 corresponds to all dimensions
parser.add_argument("--savepath", default="testi.jsonl")
parser.add_argument("--delta", default=0.001, type=float)
options = parser.parse_args()


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

print("model, tokenizer and inverter downloaded", flush=True)
if options.sentence is None:
    assert options.embedding_file is not None and options.file_index is not None
    with open(options.embedding_file, "r") as f:
        lines = f.readlines()
    if ".jsonl" in options.embedding_file:
        print("reading a jsonl", flush=True)
        d = json.loads(lines[options.file_index])
        embedding = torch.Tensor([d["emb"]]).cuda()
        try:
            options.sentence = d["text"]
        except:
           options.sentence =""
        if not os.path.isfile(options.savepath):
            try:  # try to comeup with a savepath by id
                sentence_name = d["id"]
                options.savepath = os.path.join(options.savepath, sentence_name+".jsonl")
                #print("modified", flush=True)
            except:
                raise AssertionError("--savepath not given as a .jsonl file and no identifier (id field) found on data")
    else:
        embedding = lines[options.file_index]
    del lines
else:
    embedding = get_gtr_embeddings([options.sentence], encoder, tokenizer)

print(options)
print(f"Saving to {options.savepath}")

def inversion_function(vector_to_be_inverted):
    print(f'Shape of vector-to-be-inverted {vector_to_be_inverted.shape}')
    if vector_to_be_inverted.shape[0]>8:
        results = []
        for batch in vector_to_be_inverted.split(options.batch_size): 
            r = vec2text.invert_embeddings(
                                        embeddings=batch.cuda(),
                                        num_steps=options.search_iter,
                                        corrector=corrector
                                    )
            results.extend(r)
        return results
    return vec2text.invert_embeddings(
                                        embeddings=vector_to_be_inverted.cuda(),
                                        num_steps=options.search_iter,
                                        corrector=corrector
                                    )



class Point:
    def __init__(self, loc:torch.Tensor, idx:int):
        self.idx = idx
        self.loc = loc
        self.frozen = False
        assert isinstance(self.loc, torch.Tensor), f"error in point initialisation, {type(loc), type(idx)}"
        assert isinstance(self.idx, int), f"error in point initialisation {type(loc), type(idx)}"
        self.final_inversion=None

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    @property
    def is_frozen(self):
        return self.frozen

    def move(self, delta):
        if not self.frozen:
            self.loc += delta
        else:
            return False
    
    def print_point(self):
        print(self.loc.tolist(), self.frozen)

    def invert(self):
        print(f'Trying to invert point {self.idx}')
        print(self.loc.dtype)
        return inversion_function(self.loc)

    def distance(self, reference_point):
        return eucdistance.euclidean(self.loc.squeeze().cpu(), reference_point.squeeze().cpu())


class Bubble:
    def __init__(self, origin, n_dim, n_lines, delta):
        if isinstance(origin, Point):
            assert origin.idx == 0, "Index of origin needs to be 0"
            self.origin = origin
        else:
            try:
                self.origin = Point(origin, 0)
            except:
                raise AttributeError("Give point loc as torch.Tensor")
        self.n_dim = n_dim
        self.n_lines = n_lines if n_lines > 0 else n_dim
        self.axes = self.create_axes()
        assert 0<delta<1, "Give delta-value between (0,1)."
        self.delta = delta
        self.points = self.create_initial_points()
        self.initial_inversion = self.origin.invert()[0]
        self.origin.final_inversion = self.origin.invert()[0]
        print(f'Initial inversion: {self.initial_inversion}')

    def generate_ids(self):
        """Return a list of [1..n, -1..-n]."""
        n = self.n_lines
        return list(range(1, n+1)) + list(range(-1, -n-1, -1))

    def create_axes(self):
        if self.n_lines < self.n_dim:
            if self.n_lines == 1:
                indices = np.array([0])
            else:
                indices = np.random.choice(self.n_dim, size=self.n_lines, replace=False)
            print(f'Selected axes: {indices}')
            return torch.concatenate([torch.eye(self.n_dim, dtype=torch.float32)[indices], -1.*torch.eye(self.n_dim,dtype=torch.float32)[indices]]).cuda()
        return torch.concatenate([torch.eye(self.n_dim, dtype=torch.float32), -1.*torch.eye(self.n_dim,dtype=torch.float32)]).cuda() # both directions

    def create_initial_points(self):
        ids = self.generate_ids()
        assert len(ids) == len(self.axes), "Number of ids and axes does not match"
        return [Point(self.delta*p + self.origin.loc, i) for i,p in zip(ids, self.axes)]

    def expand(self):
        for i, point in enumerate(self.points):
            point.move(self.delta*self.axes[i])

    def get_points_as_matrix(self):
        non_frozen_points = [p for p in self.points if not p.frozen]
        return torch.cat([p.loc for p in non_frozen_points], dim=0), non_frozen_points

    def invert_points(self):
        embs, ps = self.get_points_as_matrix()
        return inversion_function(embs), ps
    
    def evaluate_points(self):
        inversions, non_frozen_points = self.invert_points()
        assert len(inversions) == len(non_frozen_points), f"Inversions calculated for some frozen points(?) or some missing, inversions = {len(inversions)}, non-frozen points = {len(non_frozen_points)}"
        print("Inversions:")
        print(len(inversions), inversions)
        for i,p in enumerate(non_frozen_points):
            if inversions[i] != self.initial_inversion:
                print(f'Freezing, because {inversions[i]} != {self.initial_inversion}')
                p.final_inversion = inversions[i]
                p.freeze()
    
    def loop(self):
        while not all([p.is_frozen for p in self.points]):
            self.evaluate_points()
            self.expand()
        print("All frozen")
        self.print_everything()

    def print_distance(self):
        for p in self.points:
            print(p.idx, ": ", p.distance(self.origin.loc))

    def print_everything(self):
        with open(options.savepath, "w", encoding="utf-8") as f:
            for obj in [self.origin] + self.points:
                # Convert to dict of attributes
                data = obj.__dict__.copy()
                data["distance"] = obj.distance(self.origin.loc)
                del data["loc"]
                del data ["frozen"]
                #data["loc"] = data["loc"].cpu().tolist()
                # Write one JSON object per line
                f.write(json.dumps(data) + "\n")


bubble = Bubble(embedding, embedding.shape[1], options.n_lines, options.delta)
bubble.loop()
