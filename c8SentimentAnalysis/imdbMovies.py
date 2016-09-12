# Uses the imdb movie data set found at:
# http://ai.stanford.edu/~amaas/data/sentiment/
import numpy as np
import os
import pandas as pd
import pyprind

# Unquote and run un the code within the '''triple quotes''' on the first run
# only:
'''
# import the dataset and convert to a pandas DataFrame
pbar = pyprind.ProgBar(50000) # to track progress of the conversion
labels = { 'pos': 1, 'neg': 0 }
df = pd.DataFrame()

for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = '../data/%s/%s' %(s, l)

        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r') as infile:
                txt = infile.read()

            df = df.append([[txt, labels[l]]], ignore_index = True)
            pbar.update()

df.columns = ['review', 'sentiment']

# Original files were sorted, randomly permute:
np.random.seed(11)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('../data/movie_data.csv', index = False)
'''

# Reload from saved file
df = pd.read_csv('../data/movie_data.csv')

# Check
print df.head(3)
