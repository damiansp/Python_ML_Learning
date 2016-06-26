import numpy as np
#from scipy import exp
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform

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
