import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                          '../'))
from plot_decision_regions import plot_decision_regions


wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
    header = None)

X, y = wine.iloc[:, 1:], wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std  = sc.transform(X_test)

np.set_printoptions(precision = 4)

''' # Bug somewhere in the quoted space: ignore for now
# compute means for each wine label and for each predictor
mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis = 0))
    print('MV %s:\n%s\n' %(label, mean_vecs[label - 1]))

# compute the within-class scatter matrix
d = 13 # no. features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter

print ('Within-class scatter matrix: %sx%s' %(S_W.shape[0], S_W.shape[1]))
print ('Class label distribution: %s' %np.bincount(y_train)[1:])
# NOTE: Violates assumption of uniform distribution; rescale:

S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter

print ('Scaled within-class scatter matrix: %sx%s'
       %(S_W.shape[0], S_W.shape[1]))


mean_overall = np.mean(X_train_std, axis = 0)
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print('Between-class scatter matix: %sx%s' %(S_B.shape[0], S_B.shape[1]))
    

eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [
    (np.abs(eigen_vals[i], eigetn_vecs[:, i]) for i in range(len(eigen_vals)))]
eigen_pairs = sorted(eigen_pairs, key = lambda k: k[0], reverse = True)

print('Eigenvalues in decreasing order:\n')
for ev in eigen_pairs:
    print ev[0]
'''

# LDA in sklearn
lda = LDA(n_components = 2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)

plot_decision_regions(X_train_lda, y_train, classifier = lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc = 'lower left')
plt.show()

# On test set:
X_test_lda = lda.transform(X_test_std)

plot_decision_regions(X_test_lda, y_test, classifier = lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc = 'lower left')
plt.show()
