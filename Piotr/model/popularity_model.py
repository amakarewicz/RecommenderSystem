from some_functions import get_db
import pandas as pd

class Popularity:
    '''
    Popularity object contains list of <art limit > 
    recommended articles based on user and articles database
    :param user_id: user id
    :type arg: str
    :param articles_db: user database
    :type arg: pandas table
    :param art_limit: number of reccomended articles
    :type arg: int
    '''
    def __init__(self,user_id,articles_db,user_db=None,art_limit=5):
        self.users = user_id
        self.articles = articles_db
        self.limit=art_limit
        self.user_db=user_db
        self.reccomended_articles = None
    
    def art_head(self):
        return self.articles.head()

    def reccom(self):
        if self.user_db == None:
            selected = self.articles.sort_values(by='popularity',ascending=False).head(self.limit)[['nzz_id']]
            self.recommended = [item[0] for item in selected.values.tolist()]
        # elif user not in user_db
        # else -> bazuje na popularności autorów też
        return self.recommended