import pandas as pd
from io import StringIO

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
