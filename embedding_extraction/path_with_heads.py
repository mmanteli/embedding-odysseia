import faiss
import matplotlib.pyplot as plt
from sqlitedict import SqliteDict
from sentence_transformers import SentenceTransformer
from jsonargparse import ArgumentParser
import torch
import os
import numpy as np
from scipy.spatial import distance as eucdistance
import copy
import plotly.graph_objects as go
# distance.euclidean([1, 0, 0], [0, 1, 0]) == 1.41
try:
    from model_utils import model_name_dict, get_all_prompts, get_query
except ImportError:
    from embedding_extraction.model_utils import model_name_dict, get_all_prompts, get_query
# global
# calc_distance=None
# dist_min_limit=None


def embed(model, input_texts, options):
    """Embed with given model."""
    input_texts = [get_query(options.task, t) for t in input_texts]    # add query
    return model.encode(input_texts,
                        convert_to_tensor=False,
                        normalize_embeddings=False,
                        batch_size=options.model_batch_size)


def normalize_vectors(vectors):
    """Normalize a numpy matrix w.r.t rows."""
    if vectors.dtype is not float:
        vectors = vectors.astype('float64')
    magnitude = np.sqrt(np.einsum('...i,...i', vectors, vectors))
    return vectors / magnitude.reshape(-1,1)


def calc_distance_cosine(target, neighbors):
    """
    Calculate the cosine distance between given embeddings and the given target.
    Return the indices and distances, sorted from closest to furthest.
    """
    #print(len(target), len(neighbors))
    target = normalize_vectors(np.array(target)).reshape(-1)
    neighbors = normalize_vectors(np.array(neighbors))
    #print(target.shape)
    #print(neighbors.shape)
    distances = [np.dot(target, n) for n in neighbors]
    # print(f'Cos Distances are {distances}')
    sorted_indices = np.argsort(distances)[::-1]  # for cosine, these need to be reversed
    sorted_distances = np.sort(distances)[::-1]
    # print(f"out of them, the order of the indices is \n{sorted_indices} \nDistances \n{sorted_distances}")
    return sorted_indices, sorted_distances


def calc_distance_euclidean(target, neighbors):
    """
    Calculate the euclidean distance between given embeddings and the given target.
    Return the indices and distances, sorted from closest to furthest.
    """
    distances = [eucdistance.euclidean(target, n) for n in neighbors]
    # print(f'Euc Distances are {distances}')
    sorted_indices = np.argsort(distances)
    sorted_distances = np.sort(distances)
    # print(f"out of them, the order of the indices is \n{sorted_indices} \nDistances \n{sorted_distances}")
    return sorted_indices, sorted_distances

def get_distance_function(metric):
    """Wrap distance calculation, return correct distance funtion."""
    if metric=="cosine":
        return calc_distance_cosine
    if metric=="euclidean":
        return calc_distance_euclidean
    raise AttributeError

def get_head_with_zero_mask(vec, head, num_heads):
    """Select a range of values from a vector.
    Example: input = [1,1,1,2,4,2,3,2,3,6,5,4]
    get_head(input, 1, 3)
    >> [0,0,0,0,4,2,3,2,0,0,0,0]
    get_head(input, 0, 4)
    >> [1,1,1,0,0,0,0,0,0,0,0,0]
    number of heads starts from 0."""
    assert head < num_heads, f"given head index ({head}) must be at most {num_heads-1} for num_heads = {num_heads} ."
    assert vec.shape[-1]%num_heads==0, f"the dimension of the vector {vec.shape[-1]} is not divisible by number of heads {num_heads}"
    indices_per_head = int(vec.shape[-1]/num_heads)
    mask = np.zeros(vec.shape[-1])
    mask[indices_per_head*head:indices_per_head*(head+1)] = 1
    return mask*vec

def get_head_with_mask(vec, baseline, head, num_heads):
    """Select a range of values from a vector, and others from baseline.
    Example: input = [1,1,1,2,4,2,3,2,3,6,5,4], baseline = [0,0,0,0,0,0,0,0,0,0,0,1]
    get_head(input, baseline, 1, 3)
    >> [0,0,0,0,4,2,3,2,0,0,0,1]
    get_head(input,baseline, 0, 4)
    >> [1,1,1,0,0,0,0,0,0,0,0,1]
    number of heads starts from 0."""
    assert head < num_heads, f"given head index ({head}) must be at most {num_heads-1} for num_heads = {num_heads} ."
    assert vec.shape[-1]%num_heads==0, f"the dimension of the vector {vec.shape[-1]} is not divisible by number of heads {num_heads}"
    indices_per_head = int(vec.shape[-1]/num_heads)
    headless = copy.deepcopy(baseline)
    headless[indices_per_head*head:indices_per_head*(head+1)] = vec[-1,indices_per_head*head:indices_per_head*(head+1)]
    return headless

def get_head(vec, baseline, head, num_heads):
    assert head < num_heads, f"given head index ({head}) must be at most {num_heads-1} for num_heads = {num_heads} ."
    assert vec.shape[-1]%num_heads==0, f"the dimension of the vector {vec.shape[-1]} is not divisible by number of heads {num_heads}"
    indices_per_head = int(vec.shape[-1]/num_heads)
    return vec[-1,indices_per_head*head:indices_per_head*(head+1)], baseline[indices_per_head*head:indices_per_head*(head+1)]

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


def calculate_head_distances(point_on_line, point_comparison, num_heads, metric):
    head_distances = []
    for i in range(num_heads):
        h, h_comp = get_head(point_on_line, point_comparison, i, num_heads)
        # DO NOT RE-NORMALIZE HERE
        if metric == "cosine":
            distance = np.dot(h.reshape(-1), h_comp.reshape(-1))
        else:
            distance = eucdistance.euclidean(h.reshape(-1), h_comp.reshape(-1))
        print(f"distance for head {i} inside calc head dist: {distance}")
        head_distances.append(distance)
    return head_distances




def find_closest_distances_to_a_line(line, index, db, options):
    found_d = []
    found_i = []
    found_heads_d = []
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
        closest_N_point = points[0]
        print("neighbor shape: ", closest_N_point.shape)
        print("point is: ",point)
        found_heads_d.append(calculate_head_distances(point, closest_N_point, options.num_heads, options.metric))

    return found_d, found_i, found_heads_d

def straight_path(vec1, vec2, n=20):
    """Return a straight unidistance line between two inputs."""
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    print(vec1.shape)
    print(vec2.shape)
    assert vec1.shape == vec2.shape, "Got vectors with unmatched dimensions in straight path calculation."
    # Interpolate along the path
    return np.array([vec1 + (vec2 - vec1) * t for t in np.linspace(0, 1, n)])



def straight_pathing(options, start_text=None, target_text=None):
    """Walk a straight path and find nearest neighbors to it."""
    model = SentenceTransformer(
        model_name_dict.get(options.model, options.model),
        #prompts=get_all_prompts(),
        #default_prompt_name=options.task,
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

    start_text = "In my opinion, science should be taught in schools more" if start_text is None else start_text
    target_text = "Creme de Menthe cake" if target_text is None else target_text
    # embed them
    start_query = embed(model, [start_text], options)
    target_query = embed(model, [target_text], options)

    path = straight_path(start_query, target_query, n=options.n_step)
    print(path)

    found_d, found_i, found_heads = find_closest_distances_to_a_line(path, index, db, options)
    print(found_d)
    print(found_i)
    print(found_heads)
    plot_path = os.path.dirname(options.save_plots)
    os.makedirs(plot_path, exist_ok=True)
    plot_name = options.save_plots+f"head_distance_{options.metric}.html"
    prompt_file_name = options.save_plots+"prompts.txt"
    with open(prompt_file_name, "w") as f:
        f.write(f"--start={options.start}\n--target={options.target}\n")
    plot_distance_over_steps_plotly(found_d, filename=plot_name, section_labels=found_i, additional_y = found_heads)


def plot_distance_over_steps_old(y_values, filename="distance_plot.png", section_labels=None, additional_y=None):
    """
    Plot a distance-over-steps line chart from the provided y_values, with optional color segmentation
    based on section_labels, and saves the plot as a PNG file.

    y_values (list of float): List of distance values.
    filename (str): Output filename for the saved PNG.
    section_labels (list of str): Optional list of labels for each y-value to determine segment coloring.
                                  Must be the same length as y_values.
    """
    if section_labels and len(section_labels) != len(y_values):
        raise ValueError

    x_values = list(range(len(y_values)))
    plt.figure(figsize=(12, 6))

    if section_labels:
        # Plot each segment with color based on label
        unique_labels = list(set(section_labels))
        colors = plt.cm.get_cmap("Set3", len(unique_labels))  # up to 10 unique colors
        label_color_map = {label: colors(i) for i, label in enumerate(unique_labels)}

        for i in range(len(y_values) - 1):
            label = section_labels[i]
            plt.plot(x_values[i : i + 2], y_values[i : i + 2], color=label_color_map[label], linewidth=2)

        # Add legend
        for label, color in label_color_map.items():
            plt.plot([], [], color=color, label=label)
        plt.legend(title="Sections")
    else:
        # Single-color plot
        plt.plot(x_values, y_values, marker="o", linestyle="-", color="blue")
    if additional_y:
        additional_y = np.array(additional_y)
        for i in range(additional_y.shape[-1]):
            add_y = additional_y[:,i]
            plt.plot(x_values, add_y, linestyle=":", label = f"head {i}")

    plt.title("Distance Over Steps")
    plt.xlabel("step")
    plt.ylabel("distance")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    # plt.show()

def plot_distance_over_steps_plotly(y_values, filename="distance_plot.html", section_labels=None, additional_y=None):
    """
    Plot a distance-over-steps interactive line chart using Plotly.
    
    y_values (list of float): List of distance values.
    filename (str): Output filename for the saved HTML file.
    section_labels (list of str): Optional list of labels for each y-value to determine segment coloring.
    additional_y (np.ndarray): Optional 2D array of shape (n_steps, n_heads) for overlay lines.
    """
    if section_labels and len(section_labels) != len(y_values):
        raise ValueError("Length of section_labels must match y_values.")

    x_values = list(range(len(y_values)))
    fig = go.Figure()

    if section_labels:
        # Assign unique colors to each section label
        unique_labels = list(dict.fromkeys(section_labels))  # preserves order
        color_map = {str(label): f"hsl({(i * 360 // len(unique_labels)) % 360},70%,60%)"
                     for i, label in enumerate(unique_labels)}
        print(color_map)
        print(section_labels)

        # Draw segments with color
        for i in range(len(y_values) - 1):
            label = str(section_labels[i])
            fig.add_trace(go.Scatter(
                x=x_values[i:i+2],
                y=y_values[i:i+2],
                mode="lines",
                line=dict(color=color_map[label], width=3),
                name=label,
                legendgroup=label,
                showlegend=(i == section_labels.index(int(label))),  # show legend once per label
            ))
    else:
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            line=dict(color="blue"),
            name="Distance"
        ))

    # Add additional_y lines
    if additional_y is not None:
        additional_y = np.array(additional_y)
        for i in range(additional_y.shape[-1]):
            fig.add_trace(go.Scatter(
                x=x_values,
                y=additional_y[:, i],
                mode="lines",
                line=dict(dash="dot"),
                name=f"head {i}",
                showlegend=True,
            ))

    fig.update_layout(
        title="Distance Over Steps",
        xaxis_title="Step",
        yaxis_title="Distance",
        template="plotly_white",
        legend_title="Sections",
    )

    fig.write_html(filename)
    #fig.show()

if __name__ == "__main__":
    parser = ArgumentParser(prog="find_path.py")
    parser.add_argument("--curved", action="store_true", help="which pathing method to use")
    parser.add_argument("--straight", action="store_true", help="which pathing method to use")
    parser.add_argument("--model", type=str, default="e5", help="Model name")
    parser.add_argument("--task", default="STS", choices=["STS", "Summarization", "BitextMining", "Retrieval"],
                        help="Task (==which query to use)")
    parser.add_argument('--save_plots', type=str)
    parser.add_argument("--n_nn", type=int, default=100, help="number of nearest neighbors")
    parser.add_argument("--n_probe", type=int, default=64, help="in IVFPQ, how many neighboring cells to search")
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="cosine")
    parser.add_argument("--model_batch_size", type=int, default=32)  # tested to be the fastest out of 32 64 128
    parser.add_argument("--filled_indexer")
    parser.add_argument("--database")
    parser.add_argument("--n_iterations", "--n_iter", default=200)
    parser.add_argument("--n_step", default=100)
    parser.add_argument("--start", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--debug", type=bool, default=True, help="Verbosity etc.")

    

    options = parser.parse_args()
    options.num_heads=16
    assert bool(options.curved) != bool(options.straight), "Give only one task (--curved or --straight)"
    print(options, flush=True)
    # metric re-mapping: select distance function and set a limit for "smallest possible value" of it
    calc_distance = get_distance_function(options.metric)
    #calc_distance_cosine if options.metric == "cosine" else calc_distance_euclidean
    dist_min_limit = 1 if options.metric == "cosine" else 0  # cosine==1 <=> close, eucl==0 <=> close

    kwargs = {k: v for k, v in [("start_text", options.start), ("target_text", options.target)] if v is not None}

    if options.curved:
        pathing(options, **kwargs)
    else:
        straight_pathing(options, **kwargs)
