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

    def get_name(self):
        return self.MODEL_NAME

    def head(self,db):
        return db.head()

    @staticmethod
    def grouped_select(name,user_articles, art_db, limit):
        selected = art_db[art_db['nzz_id'].isin(user_articles)][name]    # dep. przeczytanych art
        dupl = selected.value_counts()[selected.value_counts()>1].drop(index="Unknown", errors='ignore')

        ratio = tuple(dupl)  # ratio do późniejszego wyboru
        index = list(dupl.index) # index odpowiadający ratio

        if len(ratio) == 0: #brak powtórek
            return [], 0

        recomm_for_each = []
        for item in index:
            selected = list(art_db[art_db[name] == item].sort_values(by='popularity',ascending=False) \
                    .head(limit + len(user_articles))['nzz_id'])
            # dodanie tych, które nie zostaly przeczytane
            recomm_for_each.append([item for item in selected if item not in user_articles])  
            
        # wybieram z prawdopodobiństwem (wybrane przeczytane)/(wszystkie przeczytane) artykuły
        recommended = choose_recomm(recomm_for_each,ratio,limit)
        ev = evaluation(ratio)
        return recommended, ev
    
    @staticmethod
    def select_if_no_userdb(art_db,limit):
        '''metoda recomm dla przypadku <user not in database>'''
        selected = art_db.sort_values(by='popularity',ascending=False).head(limit)[['nzz_id']]
        recommended = [item[0] for item in selected.values.tolist()]
        return recommended, 1

    @staticmethod
    def select_if_userdb(art_db, user_db, user, limit):
        '''metoda recomm dla przypadku <user in database>'''
        user_articles = user_db[user_db['user_id'] == user].iloc[:,1].tolist()
        selected = art_db.sort_values(by='popularity',ascending=False) \
                   .head(limit + len(user_articles))[['nzz_id']].values.tolist()

        recommended = [item[0] for item in selected if item[0] not in user_articles][:limit]    # wyrzucam powtórki
        return recommended, 1

    def recommend(self,user_id=1, ignored=[], limit=5):
        '''wyniki systemu rekondacji (lista <limit> wyników)'''
        if self.user_db is None:
            '''przypadek bez zaimplementowanej bazy użytkowników'''
            self.recommended, ev = self.select_if_no_userdb(self.articles_db, limit)
        elif user_id not in self.user_db.user_id.values:
            '''przypadek zaimplementowanej bazy użytkowników, użytkownik nie jest w bazie'''
            self.recommended, ev = self.select_if_no_userdb(self.articles_db, limit)
        else:
            '''przypadek zaimplementowanej bazy użytkownikow, użytkownik jest w bazie'''
            self.recommended, ev = self.select_if_userdb(self.articles_db, self.user_db, user_id, limit)

        return self.recommended, ev


class Popularity_model_author(Popularity_model):
    '''przypadek z uwzględnieniem autora'''
    MODEL_NAME = "author"

    @staticmethod
    def select_if_userdb(art_db, user_db, user, limit):
        '''metoda recomm dla przypadku <user in database>'''
        user_articles = user_db[user_db['user_id'] == user].iloc[:,1].tolist()   # artykuły przeczytane
        recommended, ev = Popularity_model.grouped_select(name='author', user_articles=user_articles, art_db=art_db, limit=limit)
        return recommended, ev


class Popularity_model_department(Popularity_model):
    '''przypadek z uwzględnieniem działu'''
    MODEL_NAME = "department"
    
    @staticmethod
    def select_if_userdb(art_db, user_db, user, limit):
        '''metoda recomm dla przypadku <user in database>'''
        user_articles = user_db[user_db['user_id'] == user].iloc[:,1].tolist()   # artykuły przeczytane
        recommended, ev = Popularity_model.grouped_select(name='department', user_articles=user_articles, art_db=art_db, limit=limit)
        return recommended, ev


class Popularity_model_merge(Popularity_model):
    '''przypadek z uwzględnieniem działu'''
    MODEL_NAME = "merged"

    @staticmethod
    def select_if_userdb(art_db, user_db, user, limit):
        '''metoda recomm dla przypadku <user in database>'''
        P, Pe = Popularity_model.select_if_userdb(art_db, user_db, user, limit)
        A, Ae = Popularity_model_author.select_if_userdb(art_db, user_db, user, limit)
        D, De = Popularity_model_department.select_if_userdb(art_db, user_db, user, limit)
        recommended = choose_recomm([P,A,D],(Pe,Ae,De),limit)
        return recommended, (Pe,Ae,De)
