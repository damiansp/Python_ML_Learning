import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA

def rbf_kernel_pca(X, gamma, n_components):
    '''
    RBF kernel PCA implementation.

    @params:
      X: {NumPy ndarray}, shape = [n_samples, n_features]
      gamma: float: Tuning parameter for RBF kernel
      n_components: int: number of principal components to return

    @return:
      X_pc: {NumPy ndarray}, shape = [n_samples, k_features]: projected data
        set
    '''

    # Calculate pairwise squared Euclidean distances in the M x N dim data set
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix
    K = np.exp(-gamma * mat_sq_dists)

    # Center kernel matrix
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtain eigenpairs from the centered kernel matrix; numpy.eigh returns
    # them in sorted order
    eigvals, eigvecs = eigh(K)

    # Collect the top k eigenvecs (projected samples)
    X_pc = np.column_stack((
        eigvecs[:, -i] for i in range(1, n_components + 1)))

    return X_pc

X, y = make_moons(n_samples = 100, random_state = 0)
plt.scatter(
    X[y == 0, 0], X[y == 0, 1], color = 'red', marker = '^', alpha = 0.5)
plt.scatter(
    X[y == 1, 0], X[y == 1, 1], color = 'blue', marker = 'o', alpha = 0.5)
plt.show()


# Project onto (standard) principal components
pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X)

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7, 3))
ax[0].scatter(X_pca[y == 0, 0],
              X_pca[y == 0, 1],
              color = 'red',
              marker = '^',
              alpha = 0.5)
ax[0].scatter(X_pca[y == 1, 0],
              X_pca[y == 1, 1],
              color = 'blue',
              marker = 'o',
              alpha = 0.5)
ax[1].scatter(X_pca[y == 0, 0],
              np.zeros((50, 1)) + 0.02,
              color = 'red',
              marker = '^',
              alpha = 0.5)
ax[1].scatter(X_pca[y == 1, 0],
              np.zeros((50, 1)) - 0.02,
              color = 'blue',
              marker = 'o',
              alpha = 0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()
