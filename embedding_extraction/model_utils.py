# this file contains mapping for model names and prompts for tasks
# separately because these are needed in multiple scripts


#------------------------------- Define aliases for models here ------------------------------- #

model_name_dict = {"e5": "intfloat/multilingual-e5-large-instruct",
                   "qwen" : "Alibaba-NLP/gte-Qwen2-7B-instruct",
                   "jina" : "jinaai/jina-embeddings-v3"
}

#-------------------------------- Define possible prompts here -------------------------------- #

# these are from https://github.com/microsoft/unilm/blob/9c0f1ff7ca53431fe47d2637dfe253643d94185b/e5/utils.py#L106
# and https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct/blob/main/scripts/eval_mteb.py
# and are used for MTEB
# One difference is that e5 uses colons!!!
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

