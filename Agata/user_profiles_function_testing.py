import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

def build_profiles(user_db, matrix, articles_db, user_id): # user_db - interactions_train

    def get_one_article_profile(item_id: str, matrix, articles_df):
        idx = articles_db['nzz_id'].tolist().index(item_id)
        item_profile = matrix[idx:idx+1]
        return item_profile

    def get_articles_profiles(ids,matrix):

        item_profiles_list = [get_one_article_profile(x,matrix, articles_db) for x in ids]
        item_profiles = scipy.sparse.vstack(item_profiles_list)
        return item_profiles

    def build_user_profile(matrix, user_id): 
        interactions = user_db[user_db['nzz_id'].isin(articles_db['nzz_id'])].set_index('user_id')
        user_item_profiles = get_articles_profiles(pd.Series(interactions.loc[user_id,'nzz_id']),matrix)
        user_profile = sklearn.preprocessing.normalize(np.sum(user_item_profiles, axis=0))
        return user_profile
    
    user_profile = build_user_profile(matrix, user_id)
    
    return user_profile