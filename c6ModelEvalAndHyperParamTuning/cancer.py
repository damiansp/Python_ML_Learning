import numpy as np
import pandas as pd
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load and preprocess data--------------------------------------------
url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/' +
       'breast-cancer-wisconsin/wdbc.data')
df = pd.read_csv(url, header = None)
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()

y = le.fit_transform(y) # changes coding from M (malignant)/B (benign) to 1/0

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.20, random_state = 2)



# Combining transformers and estimator in a pipeline------------------
# Chain feature scaling, PCA, and logistic regression in a pipeline
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components = 2)),
                    ('clf', LogisticRegression(random_state = 5))])

pipe_lr.fit(X_train, y_train)
print('Test Accuracy: %.3f' %pipe_lr.score(X_test, y_test))


# K-folds Cross validation--------------------------------------------
kfold = StratifiedKFold(y = y_train, n_folds = 10, random_state = 11)
scores = []

for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %s, Class dist: %s, Acc: %.3f'
          %(k + 1, np.bincount(y_train[train]), score))

print('CV Accuracy: %.3f +/- %.3f (SD)' %(np.mean(scores), np.std(scores)))


# Like above, but evaluate more efficiently
scores = cross_val_score(
    estimator = pipe_lr, X = X_train, y = y_train, cv = 10, n_jobs = 1)
# NOTE: n_jobs = number of processors; n_jobs = -1 means use all CPUs
print('CV Accuracy Scores: %s' %scores)
print('CV Accuracy: %.3f +/- %.3f (SD)' %(np.mean(scores), np.std(scores)))
