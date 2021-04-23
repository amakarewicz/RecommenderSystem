from some_functions import get_db, choose_recomm, evaluation
from abstract_model_class import Recommendation_model
import pandas as pd

class Popularity_model(Recommendation_model):
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
    MODEL_NAME = "popularity"

    def __init__(self, articles_db=None, user_db=None):
        super().__init__(articles_db, user_db)
        self.ev = None

    def head(self,db):
        return db.head()

    @staticmethod
    def user_articles(user_db, user_id):
        '''articles read by user'''
        user_articles = user_db[user_db['user_id'] == user_id].iloc[:,1].tolist()   
        return user_articles
    
    def recommend(self,user_id=1, limit=5,ignored=True):
        '''wyniki systemu rekondacji (lista <limit> wyników)'''
        if self.user_db is None:
            '''przypadek bez zaimplementowanej bazy użytkowników'''
            self.recommended, ev = self._select_if_no_userdb(self.articles_db, limit)
        elif user_id not in self.user_db.user_id.values:
            '''przypadek zaimplementowanej bazy użytkowników, użytkownik nie jest w bazie'''
            self.recommended, ev = self._select_if_no_userdb(self.articles_db, limit)
        else:
            '''przypadek zaimplementowanej bazy użytkownikow, użytkownik jest w bazie'''
            self.recommended, ev = self._select_if_userdb(self.articles_db,
                                                          self.user_db, user_id,
                                                          limit, ignored)
        return self.recommended, ev

    @staticmethod
    def _select_if_no_userdb(art_db,limit):
        '''metoda recommend dla przypadku <user not in database>'''
        recommended = list(art_db.sort_values(by='popularity',ascending=False).head(limit)['nzz_id'])
        return recommended, 1

    @staticmethod
    def _select_if_userdb(art_db, user_db, user_id, limit, ignored):
        '''metoda recommend dla przypadku <user in database>'''
        if ignored in [False, []]:
            # bez ignorowania
            recommended = list(art_db.sort_values(by='popularity',ascending=False).head(limit)['nzz_id'])
            return recommended, 1
        elif ignored is True:
            # ignorowanie artykułów użytkownika
            ignored = Popularity_model.user_articles(user_db, user_id)  # artykuły przeczytane
        # ignorowanie listy podanych artykułów
        selected = list(art_db.sort_values(by='popularity',ascending=False) \
                .head(limit + len(ignored))['nzz_id'])
        recommended = [item for item in selected if item not in ignored][:limit]    # wyrzucam powtórki
        
        return recommended, 1

    @staticmethod
    def key_select(name,art_db,user_articles, limit, ignored):
        '''selecting articles basing by 'groupby' name'''

        selected = art_db[art_db['nzz_id'].isin(user_articles)][name]    # dep. przeczytanych art
        dupl = selected.value_counts()[selected.value_counts()>1].drop(index="Unknown", errors='ignore')

        ratio = tuple(dupl)  # ratio do późniejszego wyboru
        index = list(dupl.index) # index odpowiadający ratio

        if len(ratio) == 0: #brak powtórek -> brak rekomendacji
            return [], 0

        if ignored in [False, []]:
            # bez ignorowania
            recomm_for_each = []
            for item in index:
                selected = list(art_db[art_db[name] == item].sort_values(by='popularity',ascending=False) \
                        .head(limit)['nzz_id'])
                # dodanie tych, które nie zostaly przeczytane
                recomm_for_each.append(selected)

        else:
            if ignored is True:
                # ignorowanie artykułów użytkownika
                ignored = user_articles

            recomm_for_each = []
            for item in index:
                selected = list(art_db[art_db[name] == item].sort_values(by='popularity',ascending=False) \
                        .head(limit + len(ignored))['nzz_id'])
                # dodanie tych, z wyłączeniem ignorowanych
                recomm_for_each.append([item for item in selected if item not in ignored]) 
        
        # wybieram z prawdopodobiństwem (wybrane przeczytane)/(wszystkie przeczytane) artykuły
        recommended = choose_recomm(recomm_for_each,ratio,limit)
        ev = evaluation(ratio)
        return recommended, ev


class Popularity_model_author(Popularity_model):
    '''przypadek z uwzględnieniem autora'''
    MODEL_NAME = "author"

    @staticmethod
    def _select_if_userdb(art_db, user_db, user_id, limit, ignored):
        '''metoda recomm dla przypadku <user in database>'''
        user_articles = Popularity_model.user_articles(user_db, user_id)
        recommended, ev = Popularity_model.key_select(name='author', art_db=art_db,
                                                      user_articles=user_articles,
                                                      limit=limit, ignored=ignored)
        return recommended, ev


class Popularity_model_department(Popularity_model):
    '''przypadek z uwzględnieniem działu'''
    MODEL_NAME = "department"
    
    @staticmethod
    def _select_if_userdb(art_db, user_db, user_id, limit, ignored):
        '''metoda recomm dla przypadku <user in database>'''
        user_articles = Popularity_model.user_articles(user_db, user_id)
        recommended, ev = Popularity_model.key_select(name='department', art_db=art_db,
                                                      user_articles=user_articles,
                                                      limit=limit, ignored=ignored)
        return recommended, ev


class Popularity_model_final(Popularity_model):
    '''przypadek z uwzględnieniem działu'''
    MODEL_NAME = "final"

    @staticmethod
    def _select_if_userdb(art_db, user_db, user_id, limit, ignored):
        '''metoda recomm dla przypadku <user in database>'''
        P, Pe = Popularity_model._select_if_userdb(art_db, user_db, user_id, limit, ignored)
        A, Ae = Popularity_model_author._select_if_userdb(art_db, user_db, user_id, limit, ignored)
        D, De = Popularity_model_department._select_if_userdb(art_db, user_db, user_id, limit, ignored)
        recommended = choose_recomm([P,A,D],(Pe,Ae,De),limit)
        return recommended, (Pe,Ae,De)
