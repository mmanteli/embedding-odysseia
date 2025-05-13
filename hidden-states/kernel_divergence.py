import numpy as np

# test cases:
# normalize (0.57, 0.483, 0.53) = (0.62225851, 0.52728221, 0.57859124)
# normalize (1,1,0) = (0.70710678, 0.70710678, 0.0)
# normalize (1,2,3) = (0.26726124, 0.53452248, 0.80178373)
# vectors = np.array([[1,2,3], [1,1,0], [0.57, 0.483, 0.53]])
# results_gold= np.array([[0.26726124, 0.53452248, 0.80178373],
#                       [0.70710678, 0.70710678, 0.0],
#                       [0.62225851, 0.52728221, 0.57859124]])
# (normalize_vectors(vectors) - results_gold < 1e-8).all()
# >> True

def normalize_vectors(vectors):
    """Normalize a numpy matrix w.r.t rows."""
    if vectors.dtype is not float:
        vectors = vectors.astype('float64')
    magnitude = np.sqrt(np.einsum('...i,...i', vectors, vectors))
    return vectors / magnitude.reshape(-1,1)

def l2_squared(vector):
    """
    Return the squared L2 norm of a vector.
    L2 norm is the inner product with the vector itself, with square root.
    """
    return np.inner(vector, vector) # no square root, as squared

def radial_basis(z, gamma, normalize=True):
    """Calculate the similarity matrix between the same set of vectors."""
    if normalize:
        normalize_vectors(z)
    result = np.zeros((z.shape[0], z.shape[0]))
    for i,v1 in enumerate(z):
        for j,v2 in enumerate(z):
            result[i,j] = np.exp(-gamma*l2_squared(v1-v2))


def kernel_divergence(phi, psi):
    """Evaluate changes between phi (baseline) and psi (target) radial basis matrises."""
    normalizer = np.sqrt(phi.sum())
    distance_matrix = np.log(phi/psi)
    combined = phi * distance_matrix
    score = combined.sum()/normalizer
    return distance_matrix, combined, score

