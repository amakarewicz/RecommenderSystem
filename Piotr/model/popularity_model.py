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
        self.user = user_id
        self.articles = articles_db
        self.limit=art_limit
        self.user_db=user_db
        self.recommended = None
    
    def head(self,db):
        return db.head()
    
    # zmienić to na class method ???
    def select_if_no_userdb(self):
        selected = self.articles.sort_values(by='popularity',ascending=False).head(self.limit)[['nzz_id']]
        self.recommended = [item[0] for item in selected.values.tolist()]
        return

    def reccom(self):
        '''wyniki systemu rekondacji'''
        if self.user_db is None:
            '''przypadek bez zaimplementowanej bazy użytkowników'''
            self.select_if_no_userdb()
        elif self.user not in self.user_db.iloc[:,0]:
            '''przypadek zaimplementowanej bazy użytkowników, użytkownik nie jest w bazie'''
            self.select_if_no_userdb()
        # else -> bazuje na popularności autorów też
            '''przypadek zaimplementowanej bazy użytkownikow, użytkownik jest w bazie'''
        return self.recommended