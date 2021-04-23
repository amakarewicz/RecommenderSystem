from abc import ABC, abstractmethod


class Recommendation_model(ABC):
    '''
    Abstract class for recommendation_models
    '''
    MODEL_NAME = "Recommendation_model"

    def __init__(self, articles_db=None, user_db=None):
        '''
        :param articles_db: database of articles, containing for each:
            [nzz_id, author, catchline, content, content_length,
             department, lead_text, pub_date, title, popularity]
        :type arg: pandas table

        :param user_db: database of users and their read articles, containg:
            [user_id, nzz_id]
        :type arg: pandas table
        '''
        self.articles_db = articles_db
        self.user_db = user_db

    def get_name(self):
        '''method get_name()
        method returning self.MODEL_NAME
        '''
        return self.MODEL_NAME

    @abstractmethod
    def recommend(self, user_id=1, limit=5, ignored=True):
        '''recommend method, returning list of <limit> ID's recommended by model

        :param user_id: user id used to find their articles in user_db
        :type arg: int

        :param limit: number of articles to recommend
        :type arg: int

        :param ignored: if ignored
                        True (default) -> articles read by user
                        list -> list of ignored articles
                        empty list / False -> not ignored
        :type arg: bool / list
        '''
        pass

    