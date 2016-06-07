import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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
