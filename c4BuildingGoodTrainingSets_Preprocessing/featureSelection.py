import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.base import clone
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler



wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
    header = None)
wine.columns = ['Class', 'Alcohol', 'Malic Acid', 'Ash', 'Alkalinity of Ash',
                'Mg', 'Total Phenols', 'Flavanoids', 'Nonflavanoid Phenols',
                'Proanthocyanins', 'Color Intensity', 'Hue',
                'OD280/OD315 of Diluted Wines', 'Proline']

X, y = wine.iloc[:, 1:].values, wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.3, random_state = 0)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train) # converts to Z scores
X_test_std = stdsc.transform(X_test)

lr = LogisticRegression(C = 0.1, penalty = 'l1') # e.g. as in Lasso regression
lr.fit(X_train_std, y_train)

print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:',     lr.score(X_test_std,  y_test))
# similar values indicate overfitting is not a concern

print lr.intercept_ # one-vs-rest, hence 3 vals
print lr.coef_


# Plot regularization paths of coeffs
fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'magenta', 'yellow', 'black', 'pink', 'lightgreen',
          'lightblue', 'grey', 'indigo', 'orange']
weights, params = [], []

for c in np.arange(-4, 6):
    lr = LogisticRegression(penalty = 'l1', C = 10 ** c, random_state = 0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10 ** c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params,
             weights[:, column],
             label = wine.columns[column + 1],
             color = color)

plt.axhline(0, color = 'black', linestyle = '--', linewidth = 3)
plt.xlim([10 ** -5, 10 ** 5])
plt.ylabel('coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc = 'upper left')
ax.legend(loc = 'upper center',
          bbox_to_anchor = (1.38, 1.03),
          ncol = 1,
          fancybox = True)
plt.show()


# Sequential Backward Selection (SBS) Algo
class SBS():
    def __init__(self,
                 estimator,
                 k_features,  # no. features desired
                 scoring = accuracy_score,
                 test_size = 0.25,
                 random_state = 1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

        
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)

        return score

        
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = self.test_size, random_state = self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(
            X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_, r = dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores) 
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]

        return self

    
    def transform(self, X):
        return X[:, self.indices_]


# Implement SBS using a knn classifier
knn = KNeighborsClassifier(n_neighbors = 4)
sbs = SBS(knn, k_features = 1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, '-o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

# Look at best 5 features
k5 = list(sbs.subsets_[8])
print(wine.columns[1:][k5])

knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train)) # 96.8
print('Test accuracy:', knn.score(X_test_std, y_test)) # 94.4 ...slight overfit

# See model with just the best 5 features
knn.fit(X_train_std[:, k5], y_train)
print('Training acc:', knn.score(X_train_std[:, k5], y_train)) # 96.0
print('Test acc:', knn.score(X_test_std[:, k5], y_test)) # 98.1 :)



# Assessing Feature Importance
# ...based on average impurity increase in a random forest
# (NOTE: this method will generally select only one among highly correlated
# features, and others will be downgraded.)
feat_labels = wine.columns[1:]
forest = RandomForestClassifier(
    n_estimators = 10000, random_state = 0, n_jobs = -1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print('%2d) %-*s %f' %(f + 1, 30, feat_labels[f], importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        color = 'cyan',
        align = 'center')
plt.xticks(range(X_train.shape[1]), feat_labels, rotation = 90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

# Reduce features based on an importance threshold
X_selected = forest.transform(X_train, threshold = 0.15)
print X_selected.shape
