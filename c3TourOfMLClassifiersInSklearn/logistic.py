from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                          '../'))
from plot_decision_regions import plot_decision_regions

def sigmoid(z):
    return 1. / (1. + np.exp(-z))


z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0.0, color = 'k')
plt.axhspan(0.0, 1.0, facecolor = '1.0', alpha = 1.0, ls = 'dotted')
plt.axhline(y = 0.5, ls = 'dotted', color = 'k')
plt.yticks([0., 0.5, 1.])
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.show()

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
(X_train, X_test, y_train, y_test) = train_test_split(
    X, y, test_size = 0.3, random_state = 0)

sc = StandardScaler()
# Scale data
sc.fit(X_train) # Use same to transform both train and test sets
X_train_std = sc.transform(X_train)
X_test_std  = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


# The param C below = 1/lambda for L2 normalization --
# as C++ -> regularization-- (= higher variance)
lr = LogisticRegression(C = 1000.0, random_state = 0)
lr.fit(X_train_std, y_train)

plot_decision_regions(
    X_combined_std, y_combined, classifier = lr, test_idx = range(105, 150))
plt.xlabel('Petal length (Z)')
plt.ylabel('Petal width (Z)')
plt.legend(loc = 'upper left')
plt.show()

# Predict probabilities
print lr.predict_proba(X_test_std[0, :]) # prob of each class


# Play with C parameter:
weights, params = [], []

for c in np.arange(-5, 5):
    lr = LogisticRegression(C = 10 ** c, random_state = 0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10 ** c)

weights = np.array(weights)
plt.plot(params, weights[:, 0], label = 'Petal length')
plt.plot(params, weights[:, 1], linestyle = '--', label = 'Petal width')
plt.ylabel('Weight coefficient')
plt.xlabel('C')
plt.legend(loc = 'upper left')
plt.xscale('log')
plt.show()

# i.e., very small C (= very large lambda), drives coeffs to 0

