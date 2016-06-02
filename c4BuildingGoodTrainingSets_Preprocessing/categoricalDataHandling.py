from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np

df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red',   'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'class']

print df

# Mapping ordinal features
# Assume sizes have natural ordering: XL = L + 1 = M + 2
size_mapping = { 'XL': 3, 'L': 2, 'M': 1 }
df['size'] = df['size'].map(size_mapping)

print '\n', df


# Encoding class labels
# (Categorical features without natural ordering)
class_mapping = {
    label : idx for idx, label in enumerate(np.unique(df['class'])) }
print '\nclass_mapping:', class_mapping

df['class'] = df['class'].map(class_mapping)
print '\n', df

# Convert numeric labels back to original strings:
inv_class_mapping = { v: k for k, v in class_mapping.items() }
df['class'] = df['class'].map(inv_class_mapping)
print '\n', df

# The same can be performed with sklearn's LabelEncoder()
class_LE = LabelEncoder()
y = class_LE.fit_transform(df['class'].values)
print '\ny:', y

# And the inverse:
y_factor = class_LE.inverse_transform(y)
print '\ny_factor:', y_factor


# One-hot coding for categorical features
X = df[['color', 'size', 'price']].values
print '\nX:\n', X

color_LE = LabelEncoder()
X[:, 0] = color_LE.fit_transform(X[:, 0])
print '\nX:\n', X
# This is typically NOT what is wanted as now the color variable will be
# treated like a continuous one.
# Instead use one-hot coding (using a dummy variable to represent presence or
# absence for each of the factors present in the variable).

ohe = OneHotEncoder(categorical_features = [0])
X = ohe.fit_transform(X).toarray()
print '\nX:\n', X

# Now the first 3 cols of X correspond to the different color values

# Alternately, and more easily:
print '\ndf:\n', df

dummy_df = pd.get_dummies(df[['price', 'color', 'size']])
print '\ndummy_df:\n', dummy_df
