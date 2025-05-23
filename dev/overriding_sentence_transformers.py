from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoConfig
import torch.nn.functional as F

model_name_dict = {"e5": "intfloat/multilingual-e5-large-instruct",
                   "qwen" : "Alibaba-NLP/gte-Qwen2-7B-instruct",
                   "jina" : "jinaai/jina-embeddings-v3",
}

#-------------------------------- Define possible prompts here -------------------------------- #

# these are from https://github.com/microsoft/unilm/blob/9c0f1ff7ca53431fe47d2637dfe253643d94185b/e5/utils.py#L106
# and https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct/blob/main/scripts/eval_mteb.py
# and are used for MTEB
# One difference is that e5 uses colons!!!
def get_task_def_by_task_name_and_type(task_type: str) -> str:
    """Get task description."""
    if task_type in ['STS']:
        return "Retrieve semantically similar text."
    if task_type in ['Summarization']:
        return "Given a news summary, retrieve other semantically similar summaries"
    if task_type in ['BitextMining']:
        return "Retrieve parallel sentences."
    if task_type in ['Retrieval']:
        return "Given a web search query, retrieve relevant passages that answer the query"
    raise ValueError

def get_all_prompts():
    """Get all tasks and names as a dict."""
    return {task: get_task_def_by_task_name_and_type(task) for task in ['STS',
                                                                        'Summarization',
                                                                        'BitextMining',
                                                                        'Retrieval']}
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


config = AutoConfig.from_pretrained(model_name_dict["e5"])
print("heads:", config.num_attention_heads)
# original sentence-transformer pipeline model
sentence_model = SentenceTransformer(
                    model_name_dict["e5"],
                    prompts=get_all_prompts(),
                    default_prompt_name="STS",
                    trust_remote_code=True,
                    device = "cuda:0" if torch.cuda.is_available() else "cpu",
                    )

sentence_model.eval()  # set to eval
tokenizer = sentence_model.tokenizer #AutoTokenizer.from_pretrained(model_name_dict["e5"])
print(sentence_model)
# Get the base transformer
transformer = sentence_model._first_module()
hf_model = transformer.auto_model
# set to eval mode
hf_model.eval()



input_texts = ["Hello World! I love you!", "Sad to see you go"]

#print("\nUsing pipeline with token_embeddings")
# SENTENCE TRANSFORMER
#result_sentence_transformer = sentence_model.encode(input_texts,
#                        output_value="token_embeddings",
#                        prompt=get_task_def_by_task_name_and_type('STS'),
#                        convert_to_tensor=True,
#                        normalize_embeddings=False)
print("\nUsing pipeline with sentence_embedding, both sentences in batch")
sentence_emb_from_sentence_transformer = sentence_model.encode(input_texts,
                        output_value="sentence_embedding",
                        prompt=get_task_def_by_task_name_and_type('STS'),
                        convert_to_tensor=True,
                        normalize_embeddings=False)

sent_emb_one_by_one = []
for t in input_texts:
    print("\nUsing pipeline with sentence_embedding, one sentence at a time")
    print(t)
    a = sentence_model.encode(t,
                            output_value="sentence_embedding",
                            prompt=get_task_def_by_task_name_and_type('STS'),
                            convert_to_tensor=True,
                            normalize_embeddings=False)
    sent_emb_one_by_one.append(a)

print("RESULT COMPARISON")
print(sentence_emb_from_sentence_transformer)
print(sent_emb_one_by_one)
exit()

# BASE HF MODEL

# modify input to have the query
#input_text = get_detailed_instruct('STS', input_text)
#print(input_text)

input_texts = [get_task_def_by_task_name_and_type('STS')+i for i in input_texts]

print(f'Inside the script: sentences to tokenise = {input_texts}')

# Hook output from the last attention layer
attention_outputs = []
def hook_fn(module, input, output):
    attention_outputs.append(output[0])

# Register the hook on last layer's attention module
hook = hf_model.encoder.layer[-1].attention.register_forward_hook(hook_fn)

inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(hf_model.device)

print(f'Inside the script: tokenized = {inputs}')
#print(f'Inside script:\ntokenized = {inputs}')
#print("INPUTS MATCH, see above")
with torch.no_grad():
    output = hf_model(**inputs, output_attentions=True)
    last_layer_attention = output.attentions[-1]
    token_embeddings = output.last_hidden_state  # [1, seq_len, hidden_size]
    attention_mask = inputs['attention_mask']    # [1, seq_len]


#print("\nUn-pooled results...?")
#print("Base model un-pooled")
#print(token_embeddings)
#print("STF token_embeddings")
#print(result_sentence_transformer)
#print("^^UNpooled outputs match")

print("\n-----------trying to pool---------------\n")

# Mean pooling by ChatGPT
def gpt_pool(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# Mean pooling from SentenceTranformers.models.Pooling.py
def STF_pool(token_embeddings, attention_mask):
    input_mask_expanded = (attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype))
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    return sum_embeddings / sum_mask


# NORMALISATION

def normalize(features):
    if type(features) is dict:
        features.update({"sentence_embedding": F.normalize(features["sentence_embedding"], p=2, dim=1)})
        return features
    return F.normalize(features, p=2, dim=1)


sentence_emb_base_model_with_gpt_code = gpt_pool(token_embeddings, attention_mask)
sentence_emb_base_mode_with_STF_pool = STF_pool(token_embeddings, attention_mask)


print("Token embeddings from base model")
print(token_embeddings)
print("Un-normalized output form base model")
print(sentence_emb_base_mode_with_STF_pool)

print("From sentence tf pipeline:")
print(sentence_emb_from_sentence_transformer)
print("pooled token emb from library rewrite:")
norm_sent_emb_base_model = normalize(sentence_emb_base_mode_with_STF_pool)
print(norm_sent_emb_base_model)
#print("pooled with ChatGPT script:")
#print(normalize(sentence_emb_base_model_with_gpt_code))
#print("----------------------")


mag_first_emb = sum([i**2 for i in norm_sent_emb_base_model[0]])
mag_second_emb = sum([i**2 for i in norm_sent_emb_base_model[1]])
print(f'magnitude of first and second with base model {mag_first_emb}, {mag_second_emb}')

mag_stf_first_emb = sum([i**2 for i in sentence_emb_from_sentence_transformer[0]])
mag_stf_second_emb = sum([i**2 for i in sentence_emb_from_sentence_transformer[1]])
print(f'magnitude of first and second with STF: {mag_stf_first_emb}, {mag_stf_second_emb}')


assert torch.allclose(sentence_emb_from_sentence_transformer, norm_sent_emb_base_model,atol=1e-4)
print("Pooling success, results match.")

hook.remove()
#print("\n----------Setting last layer to identity---------------\n")

#hf_model.encoder.layer[-1].attention.output.dense = torch.nn.Identity()
#hf_model.encoder.layer[-1].attention.output.LayerNorm = torch.nn.Identity()
#hf_model.encoder.layer[-1].intermediate.dense = torch.nn.Identity()
#hf_model.encoder.layer[-1].intermediate.intermediate_act_fn = torch.nn.Identity()
#hf_model.encoder.layer[-1].output.dense = torch.nn.Identity()
#hf_model.encoder.layer[-1].output.LayerNorm = torch.nn.Identity()

#with torch.no_grad():
#    output = hf_model(**inputs)
    #last_layer_attention = output.attentions[-1]
#    token_embeddings_but_actually_attention = output.last_hidden_state  # [1, seq_len, hidden_size]
    #attention_mask = inputs['attention_mask']

#print(result_base_model)
#print(result_sentence_transformer)
print("\n------------------Results-----------------")
print("\n-----Attention weights------\n")
print(last_layer_attention.shape)
print(last_layer_attention)
print("\n-----Attention outputs------\n")
#print(len(attention_outputs))
for h in attention_outputs:
    print(h.shape)
    print(h)

#print("\n-----Attention per head------\n")
#print(token_embeddings_but_actually_attention.shape)
#print(token_embeddings_but_actually_attention)
#print(attention_outputs[0])
#print(attention_outputs[1])
