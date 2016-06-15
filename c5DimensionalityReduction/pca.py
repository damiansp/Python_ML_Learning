import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
    header = None)

X, y = wine.iloc[:, 1:], wine.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 1)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std  = sc.transform(X_test)

# Construct covariance matrix
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('Eigenvalues\n%s' %eigen_vals)

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse = True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, 14),
        var_exp,
        alpha = 0.5,
        align = 'center',
        label = 'individual variance explained')
plt.step(range(1, 14),
         cum_var_exp,
         where = 'mid',
         label = 'cumulative variance explained')
plt.ylabel('Explained variance')
plt.xlabel('Principal components')
plt.legend(loc = 'best')
plt.show()

# Feature transformation to PC axes
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])\
               for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse = True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))

#print('Matrix W:')
#print(w)

# transformed features x' = xW;
# print(X_train_std[0].dot(w))
# do to all data points:
#print X_train_std
X_train_pca = X_train_std.dot(w)
#print X_train_pca # looks good

# Vis
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

# print y_train # looks good
# print X_train_pca[y_train == 1]

for l, c, m in zip(np.unique(y_train), colors, markers):
    print 'l:', l
    plt.scatter(X_train_pca[np.where(y_train == l), 0],
                X_train_pca[np.where(y_train == l), 1],
                c = c,
                label = l,
                marker = m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc = 'lower left')
plt.show()
               
