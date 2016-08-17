import matplotlib.pyplot as plt
#import numpy as np
#from matplotlib.ticker import FormatStrFormatter
#from scipy import exp
#from scipy.linalg import eigh
#from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_moons # make_circles
from sklearn.decomposition import  KernelPCA # PCA

X, y = make_moons(n_samples = 100, random_state = 1976)
scikit_kpca = KernelPCA(n_components = 2, kernel = 'rbf', gamma = 15)
X_skernpca = scikit_kpca.fit_transform(X)

plt.scatter(X_skernpca[y == 0, 0],
            X_skernpca[y == 0, 1],
            color = 'red',
            marker = '^',
            alpha = 0.5)
plt.scatter(X_skernpca[y == 1, 0],
            X_skernpca[y == 1, 1],
            color = 'blue',
            marker = 'o',
            alpha = 0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
