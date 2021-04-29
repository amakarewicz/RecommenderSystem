from abc import ABC, abstractmethod
import pandas as pd
from typing import Union


class Recommendation_model(ABC):
    """
    Abstract class for recommendation_models
    """
    MODEL_NAME = "Recommendation_model"

    def __init__(self, articles_db: pd.DataFrame = None, user_db: pd.DataFrame = None):
        """ init

        Args:
            articles_db (pd.DataFrame, optional): database of articles, containing for each:
                            [nzz_id, author, catchline, content, content_length,
                            department, lead_text, pub_date, title, popularity].
                            Defaults to None.
            user_db (pd.DataFrame, optional): database of users and their read articles, containg:
                            [user_id, nzz_id].
                            Defaults to None.
        """
        self.articles_db = articles_db
        self.user_db = user_db

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
        
    @abstractmethod
    def recommend(self, user_id: int, limit: int = 5, ignored: Union[list,bool] = True) -> list:
        """ recommend method, returning list of <limit> ID's recommended by model

        Args:
            user_id (int): user id used to find their articles in user_db
            limit (int, optional): number of articles to recommend. Defaults to 5.
            ignored (Union[list,bool], optional): if ignored
                                    True (default) -> articles read by user
                                    list -> list of ignored articles
                                    empty list / False -> not ignored. Defaults to True.

        Returns:
            list: list of articles recommended
        """
        pass