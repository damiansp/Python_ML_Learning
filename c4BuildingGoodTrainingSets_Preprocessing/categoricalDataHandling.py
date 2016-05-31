import pandas as pd

df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red',   'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'class']

print df
