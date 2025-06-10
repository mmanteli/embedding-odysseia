import unittest
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import sys
from transformers.utils import logging
import warnings
import os
logging.set_verbosity_error()
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TestBGEGPU(unittest.TestCase):

    def setUp(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
                    "/scratch/project_462000642/ehenriks/register-models/bge-m3-retromae/folds_improved/fold_1",
                    )
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        self.model.to("cuda:0")
        self.model.eval()
        self.input_texts = self.input_arg

    def test_batch_logits(self):

        tokenized = self.tokenizer(self.input_texts, return_tensors="pt",truncation=True,padding=True).to(self.model.device)
        in_one_batch = self.model(**tokenized).logits

        separately = []
        for t in self.input_texts:
            tokenized = self.tokenizer(t,truncation=True,return_tensors="pt").to(self.model.device)
            interm = self.model(**tokenized).logits
            separately.append(interm)

        for a, b in zip(in_one_batch, separately):
            print(a)
            print(b)
            self.assertTrue(torch.allclose(a, b,atol=1e-6), msg="Test not passed in GPU")


class TestBGECPU(unittest.TestCase):

    def setUp(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
                    "/scratch/project_462000642/ehenriks/register-models/bge-m3-retromae/folds_improved/fold_1",
                    )
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        self.model.to("cpu")
        self.model.eval()
        self.input_texts = self.input_arg

    def test_batch_logits(self):

        tokenized = self.tokenizer(self.input_texts,truncation=True,padding=True,return_tensors="pt").to(self.model.device)
        in_one_batch = self.model(**tokenized).logits

        separately = []
        for t in self.input_texts:
            tokenized = self.tokenizer(t,truncation=True,return_tensors="pt").to(self.model.device)
            interm = self.model(**tokenized).logits
            separately.append(interm)

        for a, b in zip(in_one_batch, separately):
            self.assertTrue(torch.allclose(a, b,atol=1e-6), msg="Test not passed in CPU")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("Test this in a gpu partition.")
        sys.exit()

    input_texts = ["Hello World! I love you so so much!", "Sad to see you go"]
    TestBGEGPU.input_arg = input_texts
    TestBGECPU.input_arg = input_texts

    unittest.main()
