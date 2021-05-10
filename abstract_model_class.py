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

    def __init__(self, articles_db: pd.DataFrame = None, user_db: pd.DataFrame = None,
                 matrix: pd.DataFrame = None):
        """ init

        Args:
            articles_db (pd.DataFrame, optional): database of articles, containing for each:
                            [nzz_id, author, catchline, content, content_length,
                            department, lead_text, pub_date, title, popularity].
                            Defaults to None.
            user_db (pd.DataFrame, optional): database of users and their read articles, containg:
                            [user_id, nzz_id].
                            Defaults to None.
            matrix (?????, optional): matrix with cosine_similarities of articles. 
                                      Defaults to None.
        """
        self.articles_db = articles_db
        self.user_db = user_db
        self.matrix = matrix

    def get_name(self) -> str:
        """ method returning self.MODEL_NAME
        
        Returns:
            str: model name
        """
        return self.MODEL_NAME

    @staticmethod
    def user_articles(user_db: pd.DataFrame, user_id: int) -> list:
        """ method returning list of articles read by given user

        Args:
            user_db (pd.DataFrame): list of all users interactions
            user_id (int):  user id

        Returns:
            list: list of articles read by user.
        """
        user_articles = user_db[user_db['user_id'] == user_id].iloc[:,1].tolist()   
        return user_articles

    def filter_out_similar(self, person_recs: pd.DataFrame, feature_names: np.array, model: object, article_similarity: int, 
                                                    keyword_similarity: int) -> pd.DataFrame:
        """
        method filtering out similar articles from recommendations for a given user_articles
        
        Args:
            person_recs (pd.DataFrame or list): recommendations for given user
            feature_names (np.array): elements (words) from dictionary created by vectorization
            model (object): fasttext model used for vectorization of articles' keywords
            article_similarity (int): level of similarity between articles (from (0,1) range)
            keyword_similarity (int): level of similarity between keywords (from (0,1) range)

        Returns:
            pd.DataFrame or list of filtered recommendations for given user
        """
        # flag to check at the end whether to return list or dataframe
        list_flag = False 
        # if input data is in the list, convertion to dataframe takes place
        if isinstance(person_recs, list):
            list_flag = True
            person_recs = pd.DataFrame(person_recs, columns=['nzz_id'])
        # list of recommended articles for user
        id_list = list(person_recs['nzz_id']) 
        # extracting indices of those articles (in articles_db)
        indices = self.articles_db[self.articles_db.nzz_id.isin(id_list)].index.tolist()

        # extracting pairs of similar articles based on cosine similarities between vectors
        matrix_lower = np.tril(cosine_similarity(self.matrix[indices]))
        np.fill_diagonal(matrix_lower, 0)
        similar_pairs = np.where(matrix_lower >= article_similarity)
        similar_df = pd.DataFrame(np.column_stack(similar_pairs),columns=['first_art','second_art'])

        # extracting keywords for each pair and computing similarity between them 
        if len(similar_df) >= 1:
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

                key_vec_1_sum = [0]*300
                key_vec_2_sum = [0]*300
                for j in range(5):
                    key_vec_1_sum += key_vec_1[j]
                    key_vec_2_sum += key_vec_2[j]
                
                # computing similarity
                similarity = cosine_similarity(key_vec_1_sum.reshape(1,-1), key_vec_2_sum.reshape(1,-1))
                similar_df.loc[i,'similarity'] = similarity[0][0]

            # filtering the recommendations
            new_indices = [x for i, x in enumerate(indices) if i not in list(similar_df.loc[similar_df.similarity >= keyword_similarity, 'first_art'])]
            filtered_recs_ind = self.articles_db.loc[new_indices, 'nzz_id']

            new_recs = person_recs.loc[person_recs.nzz_id.isin(list(filtered_recs_ind))]
            # different object type returned depending on input type
            if list_flag is True:
                return list(new_recs['nzz_id'])
            else:
                return new_recs
        else:
            if list_flag is True:
                return list(person_recs['nzz_id'])
            else:
                return person_recs


    @abstractmethod
    def recommend(self):
        """ recommend method, returning list of <limit> ID's recommended by model

        Args:
            user_id (int): user id used to find their articles in user_db
            limit (int, optional): number of articles to recommend. Defaults to 5.
            ignored (Union[list,bool], optional): if ignored
                                    True (default) -> articles read by user
                                    list -> list of ignored articles
                                    empty list / False -> not ignored. Defaults to True.
            articles_db (pd.DataFrame, optional): database of articles
            user_db (pd.DataFrame, optional): database of users and their read articles

        Returns:
            list: list of articles recommended
        """
        pass