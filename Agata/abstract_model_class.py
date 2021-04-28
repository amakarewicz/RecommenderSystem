from abc import ABC, abstractmethod
import pandas as pd
from typing import Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Recommendation_model(ABC):
    """
    Abstract class for recommendation_models
    """
    MODEL_NAME = "Recommendation_model"

    def __init__(self, articles_db: pd.DataFrame = None, user_db: pd.DataFrame = None, matrix: pd.DataFrame = None):
        """
        :param articles_db: database of articles, containing for each:
            [nzz_id, author, catchline, content, content_length,
             department, lead_text, pub_date, title, popularity]
        :type arg: pandas table

        :param user_db: database of users and their read articles, containg:
            [user_id, nzz_id]
        :type arg: pandas table
        """
        self.articles_db = articles_db
        self.user_db = user_db
        self.matrix = matrix

    def get_name(self) -> str:
        """ method get_name()
        method returning self.MODEL_NAME
        """
        return self.MODEL_NAME

    @staticmethod
    def user_articles(user_db: pd.DataFrame, user_id: int) -> list:
        '''method returning articles read by given user'''
        user_articles = user_db[user_db['user_id'] == user_id].iloc[:,1].tolist()   
        return user_articles

    
    def filter_out_similar(self, person_recs: pd.DataFrame, feature_names: np.array, model: object, article_similarity: int, 
                                                    keyword_similarity: int) -> pd.DataFrame:
        # TODO - filtering if person_recs is a list !!!
        # list of recommended articles for user
        id_list = list(person_recs['nzz_id']) 
        # extracting indices of those articles (in articles_db)
        indices = self.articles_db[self.articles_db.nzz_id.isin(id_list)].index.tolist()

        # extracting pairs of similar articles based on cosine similarities between vectors
        matrix_lower = np.tril(cosine_similarity(self.matrix[indices]))
        np.fill_diagonal(matrix_lower, 0)
        similar_pairs = np.where(matrix_lower>=article_similarity)
        similar_df = pd.DataFrame(np.column_stack(similar_pairs),columns=['first_art','second_art'])

        # extracting keywords for each pair and computing similrity between them 
        for i in range(len(similar_df)):
            id_1 = similar_df.loc[i,'first_art']
            sorted_1 = np.argsort(self.matrix[id_1].data)[:-(5+1):-1]
            key_1 = np.array(feature_names)[self.matrix[id_1].indices[sorted_1]]
            id_2 = similar_df.loc[i,'second_art']
            sorted_2 = np.argsort(self.matrix[id_2].data)[:-(5+1):-1]
            key_2 = np.array(feature_names)[self.matrix[id_2].indices[sorted_2]]
            
            # vectorization of keywords
            key_vec_1 = [model.get_word_vector(x) for x in key_1]
            key_vec_2 = [model.get_word_vector(x) for x in key_2]
            
            # handling occurences of empty articles
            if not key_vec_1: key_vec_1 = [model.get_word_vector('') for i in range(5)]
            if not key_vec_2: key_vec_2 = [model.get_word_vector('') for i in range(5)]
            
            # computing similarity
            cos_matrix = [[cosine_similarity(x.reshape(1,-1),y.reshape(1,-1)) for x in key_vec_1] for y in key_vec_2]
            similarity = np.mean(cos_matrix)
            similar_df['similarity'] = similarity

        # filtering the recommendations
        new_indices = [x for i, x in enumerate(indices) if i not in list(similar_df.loc[similar_df.similarity >= keyword_similarity, 'first_art'])]
        filtered_recs_ind = self.articles_db.loc[new_indices, 'nzz_id']

        new_recs = person_recs.loc[person_recs.nzz_id.isin(list(filtered_recs_ind))]
        return new_recs


    @abstractmethod
    def recommend(self, user_id: int, limit: int = 5, ignored: Union[list,bool] = True) -> list:
        """recommend method, returning list of <limit> ID's recommended by model

        :param user_id: user id used to find their articles in user_db
        :type arg: int

        :param limit: number of articles to recommend
        :type arg: int

        :param ignored: if ignored
                        True (default) -> articles read by user
                        list -> list of ignored articles
                        empty list / False -> not ignored
        :type arg: bool / list

        :return: list of articles
        :param return: list
        """
        pass