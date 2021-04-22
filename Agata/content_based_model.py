import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, items_df=None, user_profiles=None, matrix=None):
        self.item_ids = items_df['nzz_id'].tolist()
        self.items_df = items_df
        self.user_profiles = user_profiles
        self.matrix = matrix
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        #Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(self.user_profiles[person_id], self.matrix)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar items by similarity
        similar_items = sorted([(self.item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    def recommend(self, user_id, articles_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        #Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in articles_to_ignore, similar_items))

        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['nzz_id', 'recStrength']) \
                                    .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'nzz_id', 
                                                          right_on = 'nzz_id')[
                [
                    "recStrength",
                    "nzz_id",
                    "catchline",
                    "content",
                    "department",
                    "lead_text",
                    "pub_date",
                    "content_len"
                ]
            ]


        return recommendations_df