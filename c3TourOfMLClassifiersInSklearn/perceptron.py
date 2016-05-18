from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '../'))
from plot_decision_regions import plot_decision_regions

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

ppn = Perceptron(n_iter = 40, eta0 = 0.1, random_state = 0)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' %(y_test != y_pred).sum())
print('Accuracy: %.4f' %accuracy_score(y_test, y_pred))

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X = X_combined_std,
                      y = y_combined,
                      classifier = ppn,
                      test_idx = range(105, 150))
plt.xlabel('Petal length (Z)')
plt.ylabel('Petal width (Z)')
plt.legend(loc = 'upper left')
plt.show()
