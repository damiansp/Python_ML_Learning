import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer

csv_data = '''
A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
0.0,11.0,12.0,'''

csv_data = unicode(csv_data)
df = pd.read_csv(StringIO(csv_data)) # StringIO call here to mimic loading a
                                     # real .csv file

print df
print df.isnull().sum()

# df as numpy array:
print df.values


# Eliminate samples or features (rows or cols) with missing data
# Simple but (1) at the cost of loss of info; (2) can introduce bias into data
# if NAs are non-randomly distributed (as is often the case).

# drop NaN rows
print df.dropna()

# drop NaN cols
print df.dropna(axis = 1)

# drop rows only if ALL cols ar NaN
print df.dropna(how = 'all')

# drop rows with n or more NaNs
print df.dropna(thresh = 4)

# drop rows only if NaN in specific column (here: 'C')
print df.dropna(subset = ['C'])


# Imputing missing values
# Replace with (col) mean
imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imp = imp.fit(df)
imputed_data = imp.transform(df.values)
print ""
print "Imputed Data:"
print imputed_data
