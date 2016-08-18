import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

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
