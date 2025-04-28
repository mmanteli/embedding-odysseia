import faiss
from sqlitedict import SqliteDict
from sentence_transformers import SentenceTransformer
from jsonargparse import ArgumentParser
import torch
import numpy as np
from scipy.spatial import distance as eucdistance
#distance.euclidean([1, 0, 0], [0, 1, 0]) == 1.41

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
    embedded_texts = model.encode(input_texts, convert_to_tensor=False, normalize_embeddings=False, batch_size=options.model_batch_size)
    return embedded_texts


def calc_distance_cosine(target, neighbors):
    #smallest_dist = -1
    #closest_index = None
    #for i, n in enumerate(neighbors):
    #    d = np.dot(target, n)
    #    if d > smallest_dist:   # Here larger, because large values indicate closeness
    #        smallest_dist = d
    #        closest_index=i  
    distances = [np.dot(target, n) for n in neighbors]
    sorted_indices = np.argsort(distances)
    sorted_distances = np.sort(distances)
    closest_index = sorted_indices[-2]  # SECOND LARGEST, because of cosine distance, and we do not want it to return itself
    smallest_dist = distances[closest_index] 
    return closest_index, smallest_dist, sorted_indices, sorted_distances
    

def calc_distance_euclidean(target, neighbors):
    #smallest_dist = np.inf
    #closest_index = None
    #for i, n in enumerate(neighbors):
    #    d = distance.euclidean(target, n)
    #    if d < smallest_dist:    # Here smaller, because small values indicate closeness
    #        smallest_dist = d
    #        closest_index=i 
    distances = [eucdistance.euclidean(target, n) for n in neighbors]
    sorted_indices = np.argsort(distances)
    sorted_distances = np.sort(distances)
    closest_index = sorted_indices[1]  # SECOND Smallest, because of eucl distance, and we do not want it to return itself
    smallest_dist = distances[closest_index]
    return closest_index, smallest_dist, sorted_indices, sorted_distances

def get_NN(index, db, current_query, target_query, n_nn=10):
    D, I = index.search(current_query, n_nn+1) # +1 for itself
    # D is the embedding point, I is the index
    #print(I)
    neighbors = [db[str(i)]["embeddings"] for i in I[0]]
    ind, dist, all_ind, all_dist = calc_distance(target_query[0], neighbors)
    ind_sanity, dist_sanity, all_ind_sanity, all_dist_sanity = calc_distance(current_query[0], neighbors)
    #print("minimal distance to current", dist_sanity)
    #print("all distances to target:",all_dist)
    #print("all distances to current:", all_dist_sanity)
    return I[0][ind], neighbors[ind].reshape((1,-1)), dist


if __name__ == "__main__":
    parser = ArgumentParser(prog="find_path.py")
    parser.add_argument('--model',type=str,help="Model name")
    parser.add_argument('--task', default="STS", choices=["STS","Summarization","BitextMining","Retrieval"], help='Task (==which query to use)')
    parser.add_argument('--n_nn', type=int, default=50, help="number of nearest neighbors")
    parser.add_argument('--metric', choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument('--model_batch_size', type=int, default=32)  # tested to be the fastest out of 32 64 128
    parser.add_argument('--filled_indexer')
    parser.add_argument('--database')
    parser.add_argument('--debug', type=bool, default=True, help="Verbosity etc.")

    options = parser.parse_args()
    print(options, flush=True)
    model = SentenceTransformer(
                            model_name_dict.get(options.model, options.model),
                            prompts=get_all_prompts(), 
                            default_prompt_name=options.task,
                            trust_remote_code=True,
                            device = "cuda:0" if torch.cuda.is_available() else "cpu",
                            )
    if options.debug: print("Model loaded", flush=True)
    
    index = faiss.read_index(options.filled_indexer)
    index.nprobe = 64   # how many cells to search
    if options.debug: print("index loaded", flush=True)

    db = SqliteDict(options.database)
    if options.debug: print("database loaded", flush=True)

    # metric re-mapping
    calc_distance = calc_distance_cosine if options.metric=="cosine" else calc_distance_euclidean
    dist_min_limit = 1 if options.metric=="cosine" else 0

    start_text = "Christmas Turkey recipe"
    target_text = "Creme de Menthe Cake" #"It was okay."#Creme de Menthe Cake"
    start_query = embed(model, [start_text], options)
    target_query = embed(model, [target_text], options)
    #target_query_NN_ind, ind,  = get_NN(index, db, target_query, target_query, n_nn=options.n_nn)

    current_text = start_text
    current_query = start_query.copy()
    current_id=None
    found_indices = []
    max_iterations=200
    while max_iterations > 0:
        print(f"\n\nNew iteration, current {current_text} with current id {current_id}")
        ind, current_query, dist = get_NN(index, db, current_query, target_query, n_nn=options.n_nn)
        current_text = db[str(ind)]
        current_id = ind
        print(f'New current found! Distance to target: {dist}, index found {ind}: {current_text}')
        if ind not in found_indices:
            found_indices.append(ind)
        #else:
        #    print("Breaking because found the same again")
        #    break
        if abs(dist-dist_min_limit) < 1e-3:
            print("Breaking for small distance")
            break
        max_iterations -=1

    for f in found_indices:
        print(db[str(f)]["text"])



