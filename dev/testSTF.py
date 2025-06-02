import unittest
from sentence_transformers import SentenceTransformer
import torch
import sys
from transformers.utils import logging
import warnings
import os
logging.set_verbosity_error()
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TestSentenceTransformerGPU(unittest.TestCase):

    def setUp(self):
        self.model = SentenceTransformer(
                    self.model_name_arg,
                    #prompts=get_all_prompts(),
                    #default_prompt_name="STS",
                    trust_remote_code=True,
                    device = "cuda:0",
                    model_kwargs={"attn_implementation": "eager"},
                    )
        self.model.eval()
        self.input_texts = self.input_arg

    def test_sentence_embeddings(self):
        in_one_batch = self.model.encode(self.input_texts,
                        output_value="sentence_embedding",
                        #prompt="Retrieve semantically similar text.",
                        convert_to_tensor=True,
                        normalize_embeddings=False)
        separately = []
        for t in self.input_texts:
            interm = self.model.encode(t,
                        output_value="sentence_embedding",
                        #prompt="Retrieve semantically similar text.",
                        convert_to_tensor=True,
                        normalize_embeddings=False)
            separately.append(interm)

        for a, b in zip(in_one_batch, separately):
            self.assertTrue(torch.allclose(a, b,atol=1e-4), msg="Test not passed in GPU")


class TestSentenceTransformerCPU(unittest.TestCase):

    def setUp(self):
        self.model = SentenceTransformer(
                    self.model_name_arg,
                    #prompts=get_all_prompts(),
                    #default_prompt_name="STS",
                    trust_remote_code=True,
                    device = "cpu",
                    model_kwargs={"attn_implementation": "eager"},
                    )
        self.model.eval()
        self.input_texts = self.input_arg

    def test_sentence_embeddings(self):
        in_one_batch = self.model.encode(self.input_texts,
                        output_value="sentence_embedding",
                        #prompt="Retrieve semantically similar text.",
                        convert_to_tensor=True,
                        normalize_embeddings=False)
        separately = []
        for t in self.input_texts:
            interm = self.model.encode(t,
                        output_value="sentence_embedding",
                        #prompt="Retrieve semantically similar text.",
                        convert_to_tensor=True,
                        normalize_embeddings=False)
            separately.append(interm)

        for a, b in zip(in_one_batch, separately):
            self.assertTrue(torch.allclose(a, b,atol=1e-4), msg="Test not passed in CPU")


if __name__ == '__main__':
    model_name = sys.argv[1] if len(sys.argv) > 1 else ""
    if model_name == "":
        print("Usage: give model name as first parametre.")
        sys.exit()
    if not torch.cuda.is_available():
        print("Test this in a gpu partition.")
        sys.exit()
    sys.argv = sys.argv[:1] + sys.argv[2:]
    TestSentenceTransformerGPU.model_name_arg = model_name
    TestSentenceTransformerCPU.model_name_arg = model_name

    input_texts = ["Hello World! I love you!", "Sad to see you go"]
    TestSentenceTransformerGPU.input_arg = input_texts
    TestSentenceTransformerCPU.input_arg = input_texts

    unittest.main()
