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
parser.add_argument("--batch_size", default=8, type=int)
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
       ["Jack Morris is a PhD student at Cornell Tech in New York City", "It was the best of times, it was the worst of times, it was the age of wisdom,"]
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
    print(f'Shape of inverted {vector_to_be_inverted.shape}')
    if vector_to_be_inverted.shape[0]>8:
        print("\tdoing batch")
        results = []
        for batch in vector_to_be_inverted.split(options.batch_size): 
            r = vec2text.invert_embeddings(
                                        embeddings=batch.cuda(),
                                        num_steps=options.search_iter,
                                        corrector=corrector
                                    )
            results.append(r)
        return results
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
        return eucdistance.euclidean(self.loc.squeeze().cpu(), reference_point.squeeze().cpu())


class Bubble:
    def __init__(self, origin, n_dim, n_lines, delta):
        if isinstance(origin, Point):
            self.origin = origin
        else:
            try:
                self.origin = Point(origin, -1)
            except:
                raise AttributeError("Give point loc as torch.Tensor")
        self.n_dim = n_dim
        self.n_lines = n_lines
        self.axes = self.create_axes()
        assert 0<delta<1, "Give delta-value between (0,1)."
        self.delta = delta
        self.points = self.create_initial_points()
        self.initial_inversion = self.origin.invert()[0]
        print(f'Initial inversion: {self.initial_inversion}')

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
        return [Point(self.delta*p + self.origin.loc, i%self.n_lines) for i,p in enumerate(self.axes)]

    def expand(self):
        for i, point in enumerate(self.points):
            point.move(self.delta*self.axes[i])

    def get_points_as_matrix(self):
        return torch.cat([p.loc for p in self.points if not p.frozen], dim=0)

    def invert_points(self):
        return inversion_function(self.get_points_as_matrix())
    
    def evaluate_points(self):
        inversions = self.invert_points()
        print("Inversions:")
        print(len(inversions), inversions)
        for i,p in enumerate(self.points):
            if not p.frozen:
                print(f'In point {p.idx}')
                print(f'Index = {i}')
                if inversions[i] != self.initial_inversion:
                    print(f'Freezing, because {inversions[i]} != {self.initial_inversion}')
                    p.freeze()
    
    def loop(self):
        while not all([p.is_frozen for p in self.points]):
            self.evaluate_points()
            self.expand()
        print("All frozen")
        self.print_distance()

    def print_distance(self):
        for p in self.points:
            print(p.idx, ": ", p.distance(self.origin.loc))

embedding = get_gtr_embeddings([
       "Jack Morris is a PhD student at Cornell Tech in New York City"],
        encoder, tokenizer)

bubble = Bubble(embedding, embedding.shape[1], 4, 0.001)
bubble.loop()