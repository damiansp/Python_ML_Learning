# Uses the imdb movie data set found at:
# http://ai.stanford.edu/~amaas/data/sentiment/
import numpy as np
#import os
import pandas as pd
import pyprind
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

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
#print df.head(3)

count = CountVectorizer()
docs = np.array(['The sun is shining',
                 'The weather is sweet',
                 'The sun is shining and the weather is sweet'])
bag = count.fit_transform(docs)

print count.vocabulary_
print bag.toarray()

tfidf = TfidfTransformer()
np.set_printoptions(precision = 2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())


def preprocessor(text):
    # remove html markup
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)?(?:-)?(?:\)|\(|D|P)', text)
    # remove non-word chars, convert to lower, add emoticons back in
    # (minus - noses)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

# Test
print preprocessor(df.loc[0, 'review'][-50:])
print preprocessor("</a>This :) is :( a test :-)!")

df['review'] = df['review'].apply(preprocessor)

def tokenizer(text):
    return text.split()

# Ex:
sent = ('how much wood would the chucking woodchucks chuck had the ' +
        'woodchucks wanted to chuck wood')
print tokenizer(sent)

porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

print tokenizer_porter(sent)
