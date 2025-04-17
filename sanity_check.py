import faiss
from sqlitedict import SqliteDict
from sentence_transformers import SentenceTransformer
from jsonargparse import ArgumentParser
import torch

model_name_dict = {"e5": "intfloat/multilingual-e5-large-instruct",
                   "qwen" : "Alibaba-NLP/gte-Qwen2-7B-instruct",
                   "jina" : "jinaai/jina-embeddings-v3"
}
def get_task_def_by_task_name_and_type(task_type: str) -> str:
    if task_type in ['STS']:
        return "Retrieve semantically similar text."
    if task_type in ['Summarization']:
        return "Given a news summary, retrieve other semantically similar summaries"
    if task_type in ['BitextMining']:
        return "Retrieve parallel sentences."
    if task_type in ['Retrieval']:
        return "Given a web search query, retrieve relevant passages that answer the query"
    raise ValueError(f"No instruction config for task {task_type}")

def get_all_prompts():
    return {task: get_task_def_by_task_name_and_type(task) for task in ['STS', 'Summarization', 'BitextMining','Retrieval']}

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def get_query(task_name:str, text:str) -> list:
    # get the task explanation
    task = get_task_def_by_task_name_and_type(task_name)
    # combine task description and text
    return get_detailed_instruct(task, text)


def embed(model, input_texts, options):
    #input_texts = [get_query(options.task, t) for t in texts]
    #embedded_texts = model.encode(input_texts, convert_to_tensor=False, normalize_embeddings=False)
    #input_texts = texts
    embedded_texts = model.encode(input_texts, convert_to_tensor=False, normalize_embeddings=False, batch_size=options.model_batch_size)
    return embedded_texts





parser = ArgumentParser(prog="extract.py")
parser.add_argument('--model',type=str,help="Model name")
parser.add_argument('--task', default="STS", choices=["STS","Summarization","BitextMining","Retrieval"], help='Task (==which query to use)')
parser.add_argument('--model_batch_size', type=int, default=32)  # tested to be the fastest out of 32 64 128
parser.add_argument('--filled_indexer')
parser.add_argument('--database')
parser.add_argument('--debug', type=bool, default=False, help="Verbosity etc.")

options = parser.parse_args()
print(options, flush=True)

model = SentenceTransformer(
                            model_name_dict.get(options.model, options.model),
                            prompts=get_all_prompts(), 
                            default_prompt_name=options.task,
                            trust_remote_code=True,
                            device = "cuda:0" if torch.cuda.is_available() else "cpu",
                            )
print("Model loaded")
query = "It was such a great experience." #It definitely was great experience. <- in data
emb_query = embed(model, [query], options)
print("Query embedded")
index = faiss.read_index(options.filled_indexer)
print("index loaded")

D, I = index.search(emb_query, 4)
print(I[0])
db = SqliteDict(options.database)

for i in I[0]:
    print(db[str(i)])





