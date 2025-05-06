import faiss
from sqlitedict import SqliteDict
from sentence_transformers import SentenceTransformer
from jsonargparse import ArgumentParser
import torch
import numpy as np
from scipy.spatial import distance as eucdistance
#distance.euclidean([1, 0, 0], [0, 1, 0]) == 1.41
# global
calc_distance=None
dist_min_limit=None
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
    """
    Calculate the cosine distance between given embeddings and the given target.
    Return the indices and distances, sorted from closest to furthest.
    """
    distances = [np.dot(target, n) for n in neighbors]
    #print(f'Cos Distances are {distances}')
    sorted_indices = np.argsort(distances)[::-1]   # for cosine, these need to be reversed
    sorted_distances = np.sort(distances)[::-1]
    #print(f"out of them, the order of the indices is \n{sorted_indices} \nDistances \n{sorted_distances}")
    return sorted_indices, sorted_distances
    

def calc_distance_euclidean(target, neighbors):
    """
    Calculate the euclidean distance between given embeddings and the given target.
    Return the indices and distances, sorted from closest to furthest.
    """
    distances = [eucdistance.euclidean(target, n) for n in neighbors]
    #print(f'Euc Distances are {distances}')
    sorted_indices = np.argsort(distances)
    sorted_distances = np.sort(distances)
    #print(f"out of them, the order of the indices is \n{sorted_indices} \nDistances \n{sorted_distances}")
    return sorted_indices, sorted_distances

def get_NN(index, db, current_query, target_query, n_nn=10, debug=False):
    D, I = index.search(current_query, n_nn+1) # +1 for itself
    # D is the faiss coordinates (not the same as embeddings), I is the index
    # get the embeddings for these indices (I is nested so [0] for that)
    neighbors = [db[str(i)]["embeddings"] for i in I[0]]   
    sorted_ind, sorted_dist = calc_distance(target_query[0], neighbors)
    # return indices in sorted order (first closest to target), associated distances, and sorted neighbors so no recalc is needed
    indices_in_order_of_closeness_to_target = I[0][sorted_ind]
    embeddings_in_order_of_closeness_to_target = np.array(neighbors)[sorted_ind]
    if debug: assert (np.array(db[str(indices_in_order_of_closeness_to_target[0])]["embeddings"]) == embeddings_in_order_of_closeness_to_target[0]).all(), "Sorting is wacky I'm afraid, correct this."
    return I[0][sorted_ind], sorted_dist, np.array(neighbors)[sorted_ind]


def straight_path(vec1, vec2, n=10):
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    assert vec1.shape != vec2.shape, "Got vectors with unmatched dimensions in straight path calculation."
    # Interpolate along the path
    return np.array([vec1 + (vec2 - vec1) * t for t in np.linspace(0, 1, n)])


def find_path(model, index, db, start_query, target_query, target_neighbor_faiss_indices, options):
    path_indices = []
    path_distances = []
    path_points = []
    
    current_query = start_query.copy()
    past_queries = []
    max_iterations = options.n_iterations
    while max_iterations > 0:
        if options.debug: print(f"\n-------------------------\nNew iteration\n")
        closest_indices, distances, closest_points = get_NN(index, db, current_query, target_query, n_nn=options.n_nn, debug=options.debug)
        # select the index not selected yet

        # did we find a match
        match_target = [ci in target_neighbor_faiss_indices for ci in closest_indices]
        match_path = [ci in path_indices for ci in closest_indices]
        if any(match_target):  # we found one close to target
            if options.debug: print("Found nearest to target")
            return True, path_indices, path_distances, path_points
        elif all(match_path): # we only found things we have visited before
            if options.debug: print("Back tracking")
            path_indices.append(None)
            path_distances.append(None)
            path_points.append(None)
            current_query = past_queries.pop(-1)
            max_iterations -=1
            continue
        else:
            for ci, cd, cp in zip(closest_indices,  distances, closest_points):
                if ci not in path_indices:
                    if options.debug: print("Found new closest query")
                    past_queries.append(current_query)
                    path_indices.append(ci)
                    path_distances.append(cd)
                    path_points.append(cp)
                    current_query = cp.reshape(-1,1).reshape((1,-1))
                    max_iterations -=1
                    break
        
    return False, path_indices, path_distances, path_points




def pathing(options):
    # load model
    model = SentenceTransformer(
                            model_name_dict.get(options.model, options.model),
                            prompts=get_all_prompts(), 
                            default_prompt_name=options.task,
                            trust_remote_code=True,
                            device = "cuda:0" if torch.cuda.is_available() else "cpu",
                            )
    if options.debug: print("Model loaded", flush=True)
    
    index = faiss.read_index(options.filled_indexer)
    index.nprobe = options.n_probe  # how many cells to search
    if options.debug: print("Index loaded", flush=True)

    db = SqliteDict(options.database)
    if options.debug: print("Database loaded", flush=True)



    
    # Start and target
    start_text = "In my opinion, science should be taught in schools more"
    target_text = "Creme de Menthe cake" #"It was okay."#Creme de Menthe Cake"
    # embed them
    start_query = embed(model, [start_text], options)
    target_query = embed(model, [target_text], options)
    # get neighbors of target: if we reach these, we are done.
    _, target_neighbor_faiss_indices = index.search(target_query, options.n_nn+1)
    target_neighbor_faiss_indices = target_neighbor_faiss_indices[0]   # de-nesting, not selecting only one neighbor


    # finding the path
    success, path_indices, path_distances, path_points = find_path(model, index, db, start_query, target_query, target_neighbor_faiss_indices, options)
    with open('out.txt', 'w') as outfile:
        print(options, file=outfile)
        print(f"Start: {start_text}",file=outfile)
        print("\n---------------------------------\n", file=outfile)
        for f in path_indices:
            if f is None:
                print("duplicate", end = "", file=outfile)
            else:
                data_point = db[str(f)]
                print(data_point["register"], file=outfile)
                print(data_point["text"][:2500], file=outfile)
                #print(db[str(f)]["text"],file=outfile)
            print("\n---------------------------------\n", file=outfile)
        print(f"End: {target_text}",file=outfile)
        #print("\n---------------------------------\n", file=outfile)
        print("distances:\n",path_distances, file=outfile)

    # finding straight line


if __name__ == "__main__":
    parser = ArgumentParser(prog="find_path.py")
    parser.add_argument('--model',type=str,help="Model name")
    parser.add_argument('--task', default="STS", choices=["STS","Summarization","BitextMining","Retrieval"], help='Task (==which query to use)')
    parser.add_argument('--n_nn', type=int, default=100, help="number of nearest neighbors")
    parser.add_argument('--n_probe', type=int, default=64, help="in IVFPQ, how many neighboring cells to search")
    parser.add_argument('--metric', choices=["euclidean", "cosine"], default="cosine")
    parser.add_argument('--model_batch_size', type=int, default=32)  # tested to be the fastest out of 32 64 128
    parser.add_argument('--filled_indexer')
    parser.add_argument('--database')
    parser.add_argument('--n_iterations', '--n_iter', default=200)
    parser.add_argument('--debug', type=bool, default=True, help="Verbosity etc.")

    options = parser.parse_args()
    print(options, flush=True)
    # metric re-mapping: select distance function and set a limit for "smallest possible value" of it
    calc_distance = calc_distance_cosine if options.metric=="cosine" else calc_distance_euclidean
    dist_min_limit = 1 if options.metric=="cosine" else 0  # cosine==1 <=> close, eucl==0 <=> close

    pathing(options)



