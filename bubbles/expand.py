import vec2text
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import json
from pathlib import Path
from typing import Iterator, Dict, List, Union
from jsonargparse import ArgumentParser
from scipy.spatial import distance as eucdistance

parser = ArgumentParser()
parser.add_argument("--sent1", default="Jack Morris is a PhD student at Cornell Tech in New York City")
parser.add_argument("--sent2", default="It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity")
parser.add_argument("--search_iter", default=30, type=int)
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


embedding = get_gtr_embeddings([
       "Jack Morris is a PhD student at Cornell Tech in New York City",
], encoder, tokenizer)


"""
for alpha in np.arange(0.0, 1.0, 0.1):
  mixed_embedding = torch.lerp(input=embeddings[0], end=embeddings[1], weight=alpha)
  text = vec2text.invert_embeddings(
      embeddings=mixed_embedding[None].cuda(),
      corrector=corrector,
      num_steps=20,
      sequence_beam_width=4,
  )[0]
  print(f'alpha={alpha:.1f}\t', text)
"""


def inversion_function(vector_to_be_inverted):
    #if isinstance(vector_to_be_inverted, list):
    #    vector_to_be_inverted = torch.stack(vector_to_be_inverted)
    #print(vector_to_be_inverted.shape)
    #print(type(vector_to_be_inverted))
    #print(vector_to_be_inverted.dtype)
    #return ["placeholder"]
    return vec2text.invert_embeddings(
                                        embeddings=vector_to_be_inverted.cuda(),
                                        num_steps=options.search_iter,
                                        corrector=corrector
                                    )



class Point:
    def __init__(self, loc:torch.Tensor, idx:int):
        self.loc = loc
        self.frozen = False
        self.idx = idx
        assert isinstance(self.loc, torch.Tensor), f"error in point initialisation, {type(loc), type(idx)}"
        assert isinstance(self.idx, int), f"error in point initialisation {type(loc), type(idx)}"

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
        return eucdistance.euclidean(self.loc, reference_point)


class Bubble:
    def __init__(self, origin, n_dim, delta):
        if isinstance(origin, Point):
            self.origin = origin
        else:
            try:
                self.origin = Point(origin, -1)
            except:
                raise AttributeError("Give point loc as torch.Tensor")
        self.n_dim = n_dim
        self.n_lines = n_dim
        self.axes = self.create_axes()
        assert 0<delta<1, "Give delta-value between (0,1)."
        self.delta = delta
        self.points = self.create_initial_points()
        self.initial_inversion = self.origin.invert()
        print(self.initial_inversion)

    def create_axes(self):
        return torch.concatenate([torch.eye(self.n_dim, dtype=torch.float32), -1.*torch.eye(self.n_dim,dtype=torch.float32)]).cuda() # both directions
    def create_initial_points(self):
        return [Point(torch.tensor(self.delta*p + self.origin.loc, dtype=torch.float32), i) for i,p in enumerate(self.axes)]

    def expand(self):
        for i, point in enumerate(self.points):
            point.move(self.delta*self.axes[i])

    def get_points_as_matrix(self):
        return torch.stack([p.loc for p in self.points])

    def invert_points(self):
        inversions=[]
        for p in self.points:
            inversions.append(p.invert())
        return inversions
        
        #return inversion_function(self.get_points_as_matrix())
    
    def evaluate_points(self):
        inversions = self.invert_points()
        print("Inversions:")
        print(inversions)
        for i,p in self.points:
            if inversions[i] != self.initial_inversion:
                p.freeze()
    
    def loop(self):
        while not all([p.is_frozen for p in self.points]):
            self.evaluate_points()
            self.expand()
        print("All frozen")
        self.print_points()

    def print_points(self):
        for p in self.points:
            p.print()


print(embedding)
print(embedding.shape)
print(type(embedding))

bubble = Bubble(embedding, embedding.shape[1], 0.001)
bubble.loop()