import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize(data):
    stopwords_list = stopwords.words('german') 

    vectorizer = TfidfVectorizer(analyzer='word',
                        ngram_range=(1, 3), 
                        min_df=0.01,
                        max_df=0.7,
                        max_features=5000,
                        stop_words=stopwords_list)

    matrix  = vectorizer.fit_transform(data)
    feature_names = vectorizer.get_feature_names()

    return matrix, feature_names