from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))


# n_estimators = num. of individual (bootstrapped) trees
# n_jobs is for parallelizing: here 2 means 2 processors
forest = RandomForestClassifier(
    criterion = 'entropy', n_estimators = 10, random_state = 1, n_jobs = 2)

forest.fit(X_train, y_train)

plot_decision_regions(
    X_combined, y_combined, classifier = forest, test_idx = range(105, 150))
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.legend(loc = 'upper left')
plt.show()
