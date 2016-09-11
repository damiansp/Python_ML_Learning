# NOTE: as of sklearn release version (v0.17), a more sophisticaed majority
# classifier is available: sklearn.ensemble.VotingClassifier See:
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
import numpy as np
import operator
from sklearn import datasets
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.externals import six
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import _name_estimators, Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# weighting contributions example with argmax and bincount
print np.argmax(np.bincount([0, 0, 1], # predictions from 3 difft classifiers
                            weights = [0.2, 0.2, 0.6])) # conf in each clf

# Weighting with probabilty assignments to each class
ex = np.array([[0.9, 0.1],  # e.g. classifier 1 give 90% to class 0, 10% to c1
               [0.8, 0.2],  #                 2      80%             20%
               [0.4, 0.6]]) #                 3      40%             60%
p = np.average(ex, axis = 0, weights = [0.2, 0.2, 0.6]) # wt: conf in ea clf
print p             # [0.58 0.42]
print np.argmax(p)  # 0 (index)



# Implement a MajorityVoteClassifier
class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    '''
    A majority vote ensemble classifier.

    @param
    ------
    classifiers: array-like, shape = [n_classifiers]:
      Different classifiers for the ensemble
    vote: str, {'classlabel', 'probability'}
      Default: 'classlabel'
      If 'classlabel' prediction based on argmax of class labels, else 
      if 'probability', on argmax of sum of probabilities (recommended for
      calibrated classifiers).
    weights: array-like, shape = [n_classifiers]
      Optional, default: None.  If a list of int or float values are provided,
      the classifiers are weighted by importance; uniform if weights = None
    '''

    def __init__(self, classifiers, vote = 'classlabel', weights = None):
        self.classifers = classifiers
        self.named_classifiers = {
            key: value for key, value in _name_estimators(classifiers) }
        self.vote = vote
        self.weights = weights


    def fit(self, X, y):
        '''
        Fit classifiers.

        @param
        ------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
          Matrix of training samples.
        y: array-like, shape = [n_samples]
          Vector of target class labels.

        @return
        -------
        self: object
        '''

        # Use LabelEncoder to ensure class label starts with 0 (important for
        # np.argmax call in self.predict()
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []

        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)

        return self


    def predict(self, X):
        '''
        Predict class labels for X.

        @param
        ------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
          Matrix of training samples.

        @return
        -------
        maj_vote: array-like, shape = [n_samples]
          Predicted class labels.
        '''

        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis = 1)
        else: # classlabel vote
            # Collect results from clf.predict calls
            predictions = np.asarray(
                [clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights = self.weights)),
                axis = 1,
                arr = predictions)

        maj_vote = self.lablenc_.inverse_transform(maj_vote)

        return maj_vote


    def predict_proba(self, X):
        '''
        Predict class probabilities for X.

        @param
        ------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
          Matrix of training samples

        @return
        -------
        avg_proba: array-like, shape = [n_samples, n_classes]
          Weighted avg probability for each class per sample
        '''

        probas = np.asarray(
            [clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis = 0, weights = self.weights)

        return avg_proba


    def get_params(self, deep = True):
        '''Get classifier parameter names for GridSearch'''

        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep = False)
        else:
            out = self.named_classifiers.copy()

            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep = True)):
                    out['%s__%s' %(name, key)] = value

            return out



# Combining different classification algos with maj vote
# Load the iris data set
iris = datasets.load_iris()
X, y = iris.data[50:, [1,  2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

# Split into test and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.5, random_state = 4)

# Train: logistic regression, tree, k-nearest neighbors and look at performance
# based on 10-fold cv
clf1 = LogisticRegression(penalty = 'l2', C = 0.001, random_state = 7)
clf2 = DecisionTreeClassifier(
    max_depth = 1, criterion = 'entropy', random_state = 7)
clf3 = KNeighborsClassifier(n_neighbors = 1, p = 2, metric = 'minkowski')

pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])
clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']

print('10-fold cross validation:\n')

for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator = clf,
                             X = X_train,
                             y = y_train,
                             cv = 10,
                             scoring = 'roc_auc')
    print('ROC AUC: %0.2f (+/- %0.2f) [%s]'
          %(scores.mean(), scores.std(), label))


# Combine individ classifiers into a majority vote classifier
mv_clf = MajorityVoteClassifier(classifiers = [pipe1, clf2, pipe3])
clf_labels += ['Majority Voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]


for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator = clf,
                             X = X_train,
                             y = y_train,
                             cv = 10,
                             scoring = 'roc_auc')
    print('Accuracy: %0.2f (+/- %0.2f) [%s]'
          %(scores.mean(), scores.std(), label))
