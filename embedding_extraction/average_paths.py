import faiss
import matplotlib.pyplot as plt
from sqlitedict import SqliteDict
from sentence_transformers import SentenceTransformer
from jsonargparse import ArgumentParser
import torch
import os
import numpy as np
from tqdm import tqdm
try:
    from model_utils import model_name_dict, get_all_prompts
    from find_path import straight_path, embed, get_distance_function
except ImportError:
    from embedding_extraction.model_utils import model_name_dict, get_all_prompts
    from embedding_extraction.find_path import straight_path, embed, get_distance_function

def transform_line(line, translation, rotation):
    """Translate and rotate a line by given vectors."""
    # rotate:
    rot = np.linspace(-np.sin(0.25*rotation), np.sin(0.25*rotation), len(line))
    new_line = line + rot
    # translate
    return new_line + translation

def get_NN(index, db, current_query, target_query, n_nn=10, debug=False):
    """
    Get nearest neighbors (indices, distances and values) of current query, organized
    in the order of closeness to target.
    """
    D, I = index.search(current_query, n_nn + 1)  # +1 for itself
    # D is the faiss coordinates (not the same as embeddings), I is the index
    # get the embeddings for these indices (I is nested so [0] for that)
    neighbors = [db[str(i)]["embeddings"] for i in I[0]]
    sorted_ind, sorted_dist = calc_distance(target_query[0], neighbors)
    # return indices in sorted order (first closest to target), associated distances, and sorted neighbors so no recalc
    indices_in_order_of_closeness_to_target = I[0][sorted_ind]
    embeddings_in_order_of_closeness_to_target = np.array(neighbors)[sorted_ind]
    if debug:
        assert (
            np.array(db[str(indices_in_order_of_closeness_to_target[0])]["embeddings"])
            == embeddings_in_order_of_closeness_to_target[0]
        ).all(), "Sorting is wacky I'm afraid, correct this."
    return I[0][sorted_ind], sorted_dist, np.array(neighbors)[sorted_ind]

def find_closest_distances_to_a_line(line, index, options):
    found_d = []
    found_i = []
    for point in line:
        point=point.reshape((1, -1))
        indices, distances, points = get_NN(
            index,
            db,
            point,
            point,
            n_nn=0,
            debug=options.debug,
        )  # n_nn=0 because we accept values that are the same as input
        found_d.append(distances[0])
        found_i.append(indices[0])
    return found_d, found_i








if __name__ == "__main__":
    parser = ArgumentParser(prog="average_paths.py")
    parser.add_argument("--model", type=str, default="e5", help="Model name")
    parser.add_argument("--task", default="STS", choices=["STS", "Summarization", "BitextMining", "Retrieval"],
                        help="Task (==which query to use)")
    parser.add_argument('--save_plots', '--save', type=str, default="testi.png")
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="cosine")
    parser.add_argument("--filled_indexer")
    parser.add_argument("--n_probe", type=int, default=64, help="in IVFPQ, how many neighboring cells to search")
    parser.add_argument("--database")
    parser.add_argument("--n_steps", default=40)
    parser.add_argument("--n_lines", default=20)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--beta",type=float, default=0.01)
    parser.add_argument("--start", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--model_batch_size", type=int, default=32)  # tested to be the fastest out of 32 64 128
    parser.add_argument("--debug", type=bool, default=True, help="Verbosity etc.")

    options = parser.parse_args()
    calc_distance = get_distance_function(options.metric)

    model = SentenceTransformer(
        model_name_dict.get(options.model, options.model),
        prompts=get_all_prompts(),
        default_prompt_name=options.task,
        trust_remote_code=True,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    if options.debug:
        print("Model loaded", flush=True)

    index = faiss.read_index(options.filled_indexer)
    index.nprobe = options.n_probe  # how many cells to search
    if options.debug:
        print("Index loaded", flush=True)

    db = SqliteDict(options.database)
    if options.debug:
        print("Database loaded", flush=True)

    start_text = "In my opinion, science should be taught in schools more" if options.start is None else options.start
    target_text = "Creme de Menthe cake" if options.target is None else options.target
    # embed them
    start_query = embed(model, [start_text], options)
    target_query = embed(model, [target_text], options)
    if len(start_query.shape) > 1:
        start_query=start_query.reshape(start_query.shape[1])
        target_query=target_query.reshape(target_query.shape[1])
    assert start_query.shape == target_query.shape

    # get the length of used vector

    line = straight_path(start_query, target_query, n=options.n_steps)
    _, distance = calc_distance(target_query, [start_query])
    
    
    # get distance for this example:
    d_original, i_original = find_closest_distances_to_a_line(line, index, options)

    
    np.random.seed(0)
    random_translation_vec = (np.random.random((options.n_lines, start_query.shape[0]))-0.5)*options.beta
    random_rotation_vec = (2*np.pi*(np.random.random((options.n_lines, start_query.shape[0]))-0.5))*options.alpha
    #print(random_translation_vec.shape)

    final_ind = []
    final_dist = []
    for tra, rot in tqdm(zip(random_translation_vec, random_rotation_vec)):
        new_line = transform_line(line, tra, rot)
        assert new_line.shape == line.shape, f"Transformation messed things up, new line shape = {new_line.shape}"
        found_d, found_i = find_closest_distances_to_a_line(new_line, index, options)
        final_ind.append(found_i)
        final_dist.append(found_d)
    final_dist = np.array(final_dist)
    final_ind = np.array(final_ind)
    print(final_dist.shape)
    print(final_ind.shape)
    final_mean = np.mean(final_dist, axis = 0)
    final_std = np.std(final_dist, axis= 0)
    print(final_mean.shape)

    x_values = range(options.n_steps)
    y_values = final_mean
    yerr = final_std
    plt.errorbar(x_values, y_values, marker="o", linestyle="-", color="blue", yerr=yerr)
    plt.plot(x_values, np.array(d_original), color="red")
    plt.title("Distance Over Steps")
    plt.xlabel("step")
    plt.ylabel("distance")
    plt.grid(True)
    plt.tight_layout()
    ax = plt.gca()
    #ax.set_xlim([xmin, xmax])
    ax.set_ylim([0.4, 1.05])

    if "/" in options.save_plots:
        os.makedirs(os.path.dirname(options.save_plots), exist_ok=True)
    plt.savefig(options.save_plots, dpi=300)


