from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
    header = None)
wine.columns = ['Class', 'Alcohol', 'Malic Acid', 'Ash', 'Alkalinity of Ash',
               'Mg', 'Total Phenols', 'Flavanoids', 'Nonflavanoid Phenols',
               'Proanthocyanins', 'Color Intensity', 'Hue',
               'OD280/OD315 of Diluted Wines', 'Proline']
print 'Class labels:', np.unique(wine['Class'])

print 'wine.head()\n', wine.head()

X, y = wine.iloc[:, 1:].values, wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 0)

