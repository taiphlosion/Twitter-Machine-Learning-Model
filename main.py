import pandas as pd
import tasks as job
import nltk
import numpy as np

from nltk.corpus import stopwords
from string import punctuation
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression


train = pd.read_csv('train.csv')
train.dropna(inplace=True)
train = train.sample(n = 4000, random_state=69)

test = pd.read_csv('test.csv')
test.dropna(inplace=True)

train['clean_text'] = train['Text'].apply(lambda t: job.clean(t))
test['clean_text'] = test['Text'].apply(lambda t: job.clean(t))

job.bow_Classify(train, test)
job.tfidf_Classify(train,test)


