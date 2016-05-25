from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
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

tree = DecisionTreeClassifier(
    criterion = 'entropy', max_depth = 3, random_state = 0)

tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(
    X_combined, y_combined, classifier = tree, test_idx = range(105, 150))
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.legend(loc = 'upper left')
plt.show()

export_graphviz(tree,
                out_file = 'tree.dot',
                feature_names = ['petal length', 'petal width'])
