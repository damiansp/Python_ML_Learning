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
    X, y, test_size = 0.3, random_state = 0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_test)

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
