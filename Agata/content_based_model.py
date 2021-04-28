import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from abstract_model_class import Recommendation_model
from user_profiles_function import build_profiles

class ContentBasedRecommender(Recommendation_model):
    
    MODEL_NAME = 'Content-Based'
        
    def _get_similar_items_to_user_profile(self, person_id, topn=1000):

        # building 'user profile' for each user from user_db based on vectorized articles read by them
        user_profiles = build_profiles(self.user_db, self.matrix, self.articles_db)
        # list of articles ids
        item_ids = self.articles_db['nzz_id'].tolist()
        # Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(user_profiles[person_id], self.matrix)
        # Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        # Sort the similar items by similarity
        similar_items = sorted([(item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    def recommend(self, user_id, ignored=True, limit=10, return_list=True, verbose=False):
        # get similar (recommended) articles for user
        similar_items = self._get_similar_items_to_user_profile(user_id)
        if ignored is True :
            # ignoring articles already read by user
            similar_items_filtered = list(filter(lambda x: x[0] not in self.user_articles(self.user_db, user_id), similar_items))
        elif ignored in [False, []]:
            # not ignoring any articles
            similar_items_filtered = similar_items
        else:
            # ignoring articles given as parameter 'ignored'
            similar_items_filtered = list(filter(lambda x: x[0] not in ignored, similar_items))

        # return top 'limit' recommendations
        recommendations_df = [item[0] for item in similar_items_filtered][:limit]

        if not return_list:
            recommendations_df = pd.DataFrame(similar_items_filtered, columns=['nzz_id', 'recStrength']).head(limit)
        
        if verbose:
            if self.user_db is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.articles_db, how = 'left', 
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