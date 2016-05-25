from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
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

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


svm = SVC(kernel = 'linear', C = 1.0, random_state = 0)
svm.fit(X_train_std, y_train)

plot_decision_regions(
    X_combined_std, y_combined, classifier = svm, test_idx = range(105, 150))
plt.xlabel('Petal length (Z)')
plt.ylabel('Petal width (Z)')
plt.legend(loc = 'upper left')
plt.show()
