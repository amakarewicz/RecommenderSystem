import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from abstract_model_class import Recommendation_model
from user_profiles_function import build_profiles

class ContentBasedRecommender(Recommendation_model):
    
    MODEL_NAME = 'Content-Based'

    def __init__(self, articles_db: pd.DataFrame, user_db: pd.DataFrame, matrix: np.array = None):
        """
        Args:
            
        """
        super().__init__(articles_db,user_db,matrix)
        self.user_profiles = build_profiles(self.user_db, self.matrix, self.articles_db)
        
    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        """method finding articles similar to the ones read by user

        Args:
            person_id (int): user id
            topn (int, optional): how many similar articles are selected; defaults to 1000.

        Returns:
            (list): topn articles similar to user's preferences
        """
        
        # list of articles ids
        item_ids = self.articles_db['nzz_id'].tolist()
        # Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(self.user_profiles[person_id], self.matrix)
        # Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        # Sort the similar items by similarity
        similar_items = sorted([(item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    def recommend(self, user_id, articles_db = None, user_db = None, ignored=True, limit=10, ev_return=True, verbose=False):
        """recommend method, returning list of <limit> articles' ids recommended by model

        Args:
            user_id (int): user id
            ignored (Union[list,bool]): if ignored:
                                        True (default) -> articles read by user
                                        list -> list of ignored articles
                                        empty list / False -> nothing ignored
            limit (int, optional): number of recommendations returned. Defaults to 10.
            return_list (bool, optional): flag whether to return list or pd.DataFrame. Defaults to True.
            verbose (bool, optional): flag whether to return all information about article. Defaults to False.

        Raises:
            Exception: if no user_db dataframe given

        Returns:
            (list or pd.DataFrame): recommended articles
        """
        if articles_db is None: 
            articles_db = self.articles_db
        if user_db is None:
            user_db = self.user_db
        # get similar (recommended) articles for user
        similar_items = self._get_similar_items_to_user_profile(user_id)
        if ignored is True :
            # ignoring articles already read by user
            similar_items_filtered = list(filter(lambda x: x[0] not in self.user_articles(user_db, user_id), similar_items))
        elif ignored in [False, []]:
            # not ignoring any articles
            similar_items_filtered = similar_items
        else:
            # ignoring articles given as parameter 'ignored'
            similar_items_filtered = list(filter(lambda x: x[0] not in ignored, similar_items))

        # return top 'limit' recommendations
        recommendations_df = [item[0] for item in similar_items_filtered][:limit]

        if ev_return:
            recommendations_df = pd.DataFrame(similar_items_filtered, columns=['nzz_id', 'recStrength']).head(limit)
            recommendations = list(recommendations_df['nzz_id'])
            evaluations = list(recommendations_df['recStrength'])
            return recommendations_df, evaluations
        
        if verbose:
            if user_db is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(articles_db, how = 'left', 
                                                          left_on = 'nzz_id', 
                                                          right_on = 'nzz_id')[
                [
                    "recStrength",
                    "nzz_id",
                    "catchline",
                    "content",
                    "content_length",
                    "department",
                    "lead_text",
                    "pub_date",
                ]
            ]


        return recommendations_df