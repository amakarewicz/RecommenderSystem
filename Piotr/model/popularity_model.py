from some_functions import get_db
import pandas as pd

class Popularity:
    '''
    Popularity object contains list of <art limit > 
    recommended articles based on user and articles database
    :param user_id: user id
    :type arg: int
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
    
    @staticmethod
    def select_if_no_userdb(art_db,limit):
        selected = art_db.sort_values(by='popularity',ascending=False).head(limit)[['nzz_id']]
        recommended = [item[0] for item in selected.values.tolist()]
        return recommended

    @staticmethod
    def select_if_userdb(art_db, user_db, user, limit):
        user_articles = user_db[user_db['id'] == user].iloc[:,1].tolist()
        selected = art_db.sort_values(by='popularity',ascending=False) \
                   .head(limit + len(user_articles))[['nzz_id']].values.tolist()

        recommended = [item[0] for item in selected if item[0] not in user_articles][:limit]    # wyrzucam powtórki
        return recommended

    def reccom(self):
        '''wyniki systemu rekondacji'''
        if self.user_db is None:
            '''przypadek bez zaimplementowanej bazy użytkowników'''
            self.recommended = self.select_if_no_userdb(self.articles, self.limit)
        elif self.user not in self.user_db.iloc[:,0]:
            '''przypadek zaimplementowanej bazy użytkowników, użytkownik nie jest w bazie'''
            self.recommended = self.select_if_no_userdb(self.articles, self.limit)
        else:
            self.recommended = self.select_if_userdb(self.articles, self.user_db, self.user, self.limit)
            '''przypadek zaimplementowanej bazy użytkownikow, użytkownik jest w bazie'''
        return self.recommended