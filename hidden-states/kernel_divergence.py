import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
        z = normalize_vectors(z)
        assert l2_squared(z[0]) -1.0 < 1e-8, f'norm is {l2_squared(z[0])}'
    result = np.zeros((z.shape[0], z.shape[0]))
    for i,v1 in enumerate(z):
        for j,v2 in enumerate(z):
            print(v1-v2)
            print(l2_squared(v1-v2))
            result[i,j] = np.exp(-gamma*l2_squared(v1-v2))
    return result


def kernel_divergence(phi, psi):
    """Evaluate changes between phi (baseline) and psi (target) radial basis matrises."""
    normalizer = np.sqrt(phi.sum())
    distance_matrix = phi - psi #np.log(phi/psi)   # TODO this only works with gamma=1
    combined = phi * distance_matrix
    score = combined.sum()/normalizer
    return distance_matrix, combined, score

df_OP = pd.read_csv("/scratch/project_462000883/amanda/embedding-odysseia/hidden-states/testi_OP.csv")
df_LY = pd.read_csv("/scratch/project_462000883/amanda/embedding-odysseia/hidden-states/testi_LY_with_OP.csv")
df = pd.concat([df_OP, df_LY])

#vecs = np.array(df["layer_-1"].tolist())
vecs_OP = np.array([eval(i) for i in df_OP["layer_-1"]])
vecs_LY = np.array([eval(i) for i in df_LY["layer_-1"]])

vecs_OP = vecs_OP.reshape((10,-1))
vecs_LY = vecs_LY.reshape((10,-1))

phi = radial_basis(vecs_OP, 1)
psi = radial_basis(vecs_LY, 1)
print(phi)

distance, gated, value = kernel_divergence(phi, psi)
print(value)

hm0 = sns.heatmap(phi)
figure = hm0.get_figure()
figure.savefig('base.png', dpi=400)
plt.clf()

hm1 = sns.heatmap(distance)
figure = hm1.get_figure()
figure.savefig('distance.png', dpi=400)
plt.clf()

hm2 = sns.heatmap(gated)
figure = hm2.get_figure()
figure.savefig('gated.png', dpi=400)
