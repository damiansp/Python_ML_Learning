import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve, validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, f1_score, make_scorer, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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

pipe_lr = Pipeline([
    ('scl', StandardScaler()),
    ('clf', LogisticRegression(penalty = 'l2', random_state = 7))])

train_sizes, train_scores, test_scores = learning_curve(
    estimator = pipe_lr,
    X = X_train,
    y = y_train,
    train_sizes = np.linspace(0.1, 1.0, 10),
    cv = 10, n_jobs = 1)
train_mean = np.mean(train_scores, axis = 1)
train_std  = np.std(train_scores,  axis = 1)
test_mean  = np.mean(test_scores,  axis = 1)
test_std   = np.std(test_scores,   axis = 1)

plt.plot(train_sizes,
         train_mean,
         color = 'blue',
         marker = 'o',
         markersize = 5,
         label = 'training accuracy')
plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha = 0.15,
                 color = 'blue')
plt.plot(train_sizes,
         test_mean,
         color = 'red',
         linestyle = '--',
         marker = 's',
         markersize = 5,
         label = 'validation accuracy')
plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha = 0.15,
                 color = 'red')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right')
plt.ylim([0.8, 1.0])
plt.show()


# Adressing Over/Underfitting with Validation Curves------------------
param_range = [0.001, 0.01, 0.1, 1., 10., 100.]
train_scores, test_scores = validation_curve(estimator = pipe_lr,
                                             X = X_train,
                                             y = y_train,
                                             param_name = 'clf__C',
                                             param_range = param_range,
                                             cv = 10)
train_mean = np.mean(train_scores, axis = 1)
train_std  = np.std(train_scores,  axis = 1)
test_mean  = np.mean(test_scores,  axis = 1)
test_std   = np.std(test_scores,   axis = 1)

plt.plot(param_range,
         train_mean,
         color = 'blue',
         marker = 'o',
         markersize = 5,
         label = 'training accuracy')
plt.fill_between(param_range,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha = 0.15,
                 color = 'blue')
plt.plot(param_range,
         test_mean,
         color = 'red',
         linestyle = '--',
         marker = 's',
         markersize = 5,
         label = 'validation accuracy')
plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha = 0.15,
                 color = 'red')
plt.grid()
plt.xscale('log')
plt.legend(loc = 'lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.show()



# Grid Search---------------------------------------------------------
pipe_svc = Pipeline([('scl', StandardScaler()),
                     ('clf', SVC(random_state = 6))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1., 10., 100., 1000.]
param_grid = [{ 'clf__C': param_range,
                'clf__kernel': ['linear'] },
              { 'clf__C': param_range,
                'clf__gamma': param_range,
                'clf__kernel': ['rbf'] }]
gs = GridSearchCV(estimator = pipe_svc,
                  param_grid = param_grid,
                  scoring = 'accuracy',
                  cv = 10,
                  n_jobs = -1)
gs = gs.fit(X_train, y_train)

print(gs.best_score_)
print(gs.best_params_)

# Use best params on test data
clf = gs.best_estimator_
clf.fit(X_train, y_train)

print('Test accuracy (SVM): %.3f' %clf.score(X_test, y_test))



# Nested Cross-Validation---------------------------------------------
gs = GridSearchCV(estimator = pipe_svc,
                  param_grid = param_grid,
                  scoring = 'accuracy',
                  cv = 5,
                  n_jobs = -1)
scores = cross_val_score(gs, X, y, scoring = 'accuracy', cv = 5)

print('CV accuracy (SVM): %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

# Compare with a tuned tree mod
gs = GridSearchCV(estimator = DecisionTreeClassifier(random_state = 9),
                  param_grid = [{ 'max_depth': [1, 2, 3, 4, 5, 6, 7, None] }],
                  scoring = 'accuracy',
                  cv = 5)
scores = cross_val_score(gs, X_train, y_train, scoring = 'accuracy', cv = 5)

print ('CV accuracy (tree): %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))



# Performance Evaluation Metrics--------------------------------------
pipe_svc.fit(X_train, y_train)

y_pred = pipe_svc.predict(X_test)
conf_mat = confusion_matrix(y_true = y_test, y_pred = y_pred)
print(conf_mat)



# Optimizing Precision and Recall of a Classification Model-----------
print('Precision: %.3f' %precision_score(y_true = y_test, y_pred = y_pred))
print('Recall:    %.3f' %recall_score(   y_true = y_test, y_pred = y_pred))
print('F1:        %.3f' %f1_score(       y_true = y_test, y_pred = y_pred))

# Change metrics so label 0 is the "positive" case, and assign metric for
# validation in grid search
scorer = make_scorer(f1_score, pos_label = 0)
gs = GridSearchCV(
    estimator = pipe_scv, param_grid = param_grid, scoring = scorer, cv = 10)
# ...
