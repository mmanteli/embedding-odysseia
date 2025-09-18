import vec2text
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import json
from pathlib import Path
from typing import Iterator, Dict, List, Union
from jsonargparse import ArgumentParser

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

class Point:
    def __init__(self, loc:torch.Tensor):
        self.loc = loc
        self.frozen = False
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
    
    def print(self):
        print(self.loc)

    def invert(self):
        return vec2text.invert_embeddings(
                                        embeddings=self.loc.cuda(),
                                        num_steps=options.search_iter,
                                        corrector=corrector
                                    )


class Bubble:
    def __init__(self, origin, n_dim, delta):
        if isinstance(origin, Point):
            self.origin = origin
        else:
            try:
                self.origin = Point(origin)
            except:
                print("Someting wrong")
        self.n_dim = n_dim
        self.n_lines = n_dim
        self.axes = self.create_lines()
        assert 0<delta<1, "Give delta-value between (0,1)."
        self.delta = delta
        self.points = self.create_initial_points()
        self.initial_inversion = self.origin.invert()
        print(self.initial_inversion)

    def create_lines(self):
        return torch.concatenate([torch.eye(self.n_dim, dtype=float), -1.*torch.eye(self.n_dim,dtype=float)]) # both directions
    def create_initial_points(self):
        return [Point(self.delta*p) for p in self.axes]

    def expand(self):
        for i, point in enumerate(self.points):
            point.move(self.delta*self.axes[i])
    def invert_points(self):
        return vec2text.invert_embeddings(
                                        embeddings=self.points.cuda(),
                                        num_steps=options.search_iter,
                                        corrector=corrector
                                    )
    def evaluate_points(self):
        inversions = self.invert_points()
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