from some_functions import get_db, choose_recomm, evaluation
import pandas as pd

class Popularity_model:

    def __init__(self,articles_db,user_db=None):
        # model przystosowany do testowania Kamila, user w recomm, nie w init
        self.articles = articles_db
        self.user_db=user_db
        self.recommended = []
    
    def head(self,db):
        return db.head()
    
    @staticmethod
    def select_if_no_userdb(art_db,limit):
        '''metoda recomm dla przypadku <user not in database>'''
        selected = art_db.sort_values(by='popularity',ascending=False).head(limit)[['nzz_id']]
        recommended = [item[0] for item in selected.values.tolist()]
        return recommended

    @staticmethod
    def select_if_userdb(art_db, user_db, user, limit):
        '''metoda recomm dla przypadku <user in database>'''
        user_articles = user_db[user_db['id'] == user].iloc[:,1].tolist()
        selected = art_db.sort_values(by='popularity',ascending=False) \
                   .head(limit + len(user_articles))[['nzz_id']].values.tolist()

        recommended = [item[0] for item in selected][:limit]    # wyrzucam powtórki
        return recommended

    def recomm(self, user, limit):
        '''wyniki systemu rekondacji (lista <limit> wyników)'''
        if self.user_db is None:
            '''przypadek bez zaimplementowanej bazy użytkowników'''
            self.recommended = self.select_if_no_userdb(self.articles, limit)
        elif user not in self.user_db.id.values:
            '''przypadek zaimplementowanej bazy użytkowników, użytkownik nie jest w bazie'''
            self.recommended = self.select_if_no_userdb(self.articles, limit)
        else:
            '''przypadek zaimplementowanej bazy użytkownikow, użytkownik jest w bazie'''
            self.recommended = self.select_if_userdb(self.articles, self.user_db, user, limit)
        db = pd.DataFrame(self.recommended,columns=['nzz_id'])
        return db


class Popularity_model_author(Popularity_model):
    '''przypadek z uwzględnieniem autora'''
 
    @staticmethod
    def select_if_userdb(art_db, user_db, user, limit):
        '''metoda recomm dla przypadku <user in database>'''
        user_articles = user_db[user_db['id'] == user].iloc[:,1].tolist()   # artykuły przeczytane
        authors = art_db[art_db['nzz_id'].isin(user_articles)]['author']    # dep. przeczytanych art
        dupl = authors.value_counts()[authors.value_counts()>1].drop(index="Unknown", errors='ignore')

        ratio = tuple(dupl)  # ratio do późniejszego wyboru
        index = list(dupl.index) # index odpowiadający ratio

        if len(ratio) == 0: #brak powtórek
            return []

        recomm_for_each = []
        for item in index:
            selected = list(art_db[art_db['author'] == item].sort_values(by='popularity',ascending=False) \
                    .head(limit + len(user_articles))['nzz_id'])
            # dodanie tych, które nie zostaly przeczytane
            recomm_for_each.append([item for item in selected])  
            
        # wybieram z prawdopodobiństwem (wybrane przeczytane)/(wszystkie przeczytane) artykuły
        recommended = choose_recomm(recomm_for_each,ratio,limit)
        # ev = evaluation(ratio)
        return recommended


class Popularity_model_department(Popularity_model):
    '''przypadek z uwzględnieniem działu'''
 
    @staticmethod
    def select_if_userdb(art_db, user_db, user, limit):
        '''metoda recomm dla przypadku <user in database>'''
        user_articles = user_db[user_db['id'] == user].iloc[:,1].tolist()   # artykuły przeczytane
        departs = art_db[art_db['nzz_id'].isin(user_articles)]['department']    # dep. przeczytanych art
        dupl = departs.value_counts()[departs.value_counts()>1].drop(index="Unknown", errors='ignore')
        
        ratio = tuple(dupl)  # ratio do późniejszego wyboru
        index = list(dupl.index) # index odpowiadający ratio
        # print(dupl)
        if len(ratio) == 0: #brak powtarzających się schematów
            return []
        recomm_for_each = []
        for item in index:
            selected = list(art_db[art_db['department'] == item].sort_values(by='popularity',ascending=False) \
                    .head(limit + len(user_articles))['nzz_id'])
            # dodanie tych, które nie zostaly przeczytane
            recomm_for_each.append([item for item in selected])  
        # wybieram z prawdopodobiństwem (wybrane przeczytane)/(wszystkie przeczytane) artykuły
        recommended = choose_recomm(recomm_for_each,ratio,limit)
        # ev = evaluation(ratio)
        return recommended
