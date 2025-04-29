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

def get_NN(index, db, current_query, target_query, n_nn=10):
    D, I = index.search(current_query, n_nn+1) # +1 for itself
    # D is the faiss coordinates (not the same as embeddings), I is the index
    # get the embeddings for these indices (I is nested so [0] for that)
    neighbors = [db[str(i)]["embeddings"] for i in I[0]]   
    sorted_ind, sorted_dist = calc_distance(target_query[0], neighbors)
    sorted_ind_sanity, sorted_dist_sanity = calc_distance(current_query[0], neighbors)
    #print(f"Beginnnings of sorted distance to target: {sorted_dist}\n Beginnings of sorted distance to current {sorted_dist_sanity}")
    #print("minimal distance to current", dist_sanity)
    #print("all distances to target:",all_dist)
    #print("all distances to current:", all_dist_sanity)
    #return I[0][ind], neighbors[ind].reshape((1,-1)), dist
    return I[0], sorted_ind, sorted_dist, neighbors


if __name__ == "__main__":
    parser = ArgumentParser(prog="find_path.py")
    parser.add_argument('--model',type=str,help="Model name")
    parser.add_argument('--task', default="STS", choices=["STS","Summarization","BitextMining","Retrieval"], help='Task (==which query to use)')
    parser.add_argument('--n_nn', type=int, default=100, help="number of nearest neighbors")
    parser.add_argument('--metric', choices=["euclidean", "cosine"], default="cosine")
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

    start_text = "In my opinion, science should be taught in schools more"
    target_text = "Creme de Menthe cake" #"It was okay."#Creme de Menthe Cake"
    start_query = embed(model, [start_text], options)
    target_query = embed(model, [target_text], options)
    _, target_neighbor_faiss_indices = index.search(target_query, options.n_nn+1)
    target_neighbor_faiss_indices = target_neighbor_faiss_indices[0]
    print("Indices closest to target are: ", target_neighbor_faiss_indices)

    current_text = start_text
    current_query = start_query.copy()
    current_id=None
    found_indices = []
    found_distances = []
    max_iterations=20
    while max_iterations > 0:
        print(f"\n-------------------------\nNew iteration: \n{current_id} \n{current_text} \n")
        faiss_indices, sorted_indices, sorted_distances, embedded_neighbors = get_NN(index, db, current_query, target_query, n_nn=options.n_nn)
        #print(f'Found faiss indices are {faiss_indices}\n')
        #for f in faiss_indices:
        #    print(f"\t{f}: {db[str(f)]}")
        current_query = embedded_neighbors[sorted_indices[1]].reshape((1,-1)) # select second closest embedding
        #print(f'From above faiss indices, selecting index {sorted_indices[1]}, as it is the second closest')
        ind = faiss_indices[sorted_indices[1]] # Second closest index corresponts to it
        #print(f'Selected index is {ind}')
        dist = sorted_distances[1] # THIS TOO # second closest distance
        #print(f'and it corresponds to distance {dist}')
        current_text = db[str(ind)]["text"]
        current_id = ind
        #print(f'Selected {ind}: {current_text}, distance to target: {dist}, ')
        if ind not in found_indices:
            found_indices.append(ind)
            found_distances.append(dist)
        elif ind in target_neighbor_faiss_indices:
            print("FOUND NEAREST!!!")
            found_indices.append(ind)
            found_distances.append(dist)
            break
        else:
            print("...Found the same again")
            found_indices.append(None)
            found_distances.append(found_distances[-1])
        #    break
        if abs(dist-dist_min_limit) < 1e-3:
            print("Breaking for small distance")
            found_indices.append(ind)
            found_distances.append(dist)
            break
        max_iterations -=1

    with open('out.txt', 'w') as outfile:
        print(options, file=outfile)
        print(f"Start: {start_text}",file=outfile)
        print("\n---------------------------------\n", file=outfile)
        for f in found_indices:
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
        print("distances:\n",found_distances, file=outfile)



