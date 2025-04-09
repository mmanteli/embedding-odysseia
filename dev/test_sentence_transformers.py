from sentence_transformers import SentenceTransformer
import seaborn as sns
import torch

# this is modified from https://huggingface.co/intfloat/multilingual-e5-large-instruct examples

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


# from https://github.com/microsoft/unilm/blob/9c0f1ff7ca53431fe47d2637dfe253643d94185b/e5/utils.py#L106
def get_task_def_by_task_name_and_type(task_type: str) -> str:
    if task_type in ['STS']:
        return "Retrieve semantically similar text."

    if task_type in ['Summarization']:
        return "Given a summary, retrieve other semantically similar summaries"

    if task_type in ['BitextMining']:
        return "Retrieve parallel sentences."
    
    if task_type in ['Retrieval']:
        return "Given a web search query, retrieve relevant passages that answer the query"

    raise ValueError(f"No instruction config for task {task_type}")


def get_queries_and_documents(task_name:str) -> list:
    # get the task explanation
    task = get_task_def_by_task_name_and_type(task_name)

    if task_name in ['Retrieval']:
        queries = [
            get_detailed_instruct(task, 'how much protein should a female eat'),
            get_detailed_instruct(task, '南瓜的家常做法')
        ]

        documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"
        ]
        return queries, documents

    if task_name in ["BitextMining", "STS"]:
        # these are from hplt v2 parallel corpus 
        queries = []
        with open("bitextdata/bitextmining_examples_en.txt", "r") as f:
            for line in f.readlines():
                queries.append(get_detailed_instruct(task, line))
        with open("bitextdata/bitextmining_examples_fi.txt", "r") as f:
            documents = f.readlines()
        return queries, documents
    
    raise ValueError(f"No query and document config for task {task_name}")



task_name = "BitextMining"
model_name = "e5"


queries, documents = get_queries_and_documents(task_name)
input_texts = queries + documents

model_name_dict = {"e5": "intfloat/multilingual-e5-large-instruct",
                   "qwen" : "Alibaba-NLP/gte-Qwen2-7B-instruct",
                   "jina" : "jinaai/jina-embeddings-v3"
}
model = SentenceTransformer(model_name_dict[model_name],trust_remote_code=True)
#model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True)
# In case you want to reduce the maximum length:
if model_name == "qwen":
    model.max_seq_length = 8192

embeddings = model.encode(input_texts, convert_to_tensor=True, normalize_embeddings=True)
#torch.save(embeddings, f'{model_name}_{task_name}.pt')
print(f'embedding size {embeddings.size()}')
num_queries = len(queries)
scores = (embeddings[:num_queries] @ embeddings[num_queries:].T) * 100
fig = sns.heatmap(scores,annot=True, fmt=".1f")
figure = fig.get_figure()    
figure.savefig(f'{model_name}_heatmap_{task_name}.png')
print(scores.tolist())