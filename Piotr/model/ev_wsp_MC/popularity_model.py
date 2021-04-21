from some_functions import get_db, prob_vector_from_ratio
import pandas as pd
from math import sqrt
import numpy as np

def evaluation(ratio):
    # sqrt(sum(r^2))
    se = sqrt(sum( [(it - 1)**2 for it in ratio]))
    return round(se,2)

def choose_recomm(models_recommendations,ratio,limit,w=(1,1,1)):
    ''' dla wybranych rekomendacji z modeli wybieramy wg prawdopodobieństwa wyniki z każdego setu
    :par models_recommendations: lista z listami rekomendacji [[r11,r12,r13],[r21,r22,r23],...]
    :par ratio: stosunek wagi poszczegolnych modeli
    '''
    # nadanie wagi
    if len(ratio) == 3:
        ratio = np.array(ratio)*np.array(w)
    if len(models_recommendations) != len(ratio):
        raise ValueError
    vect = prob_vector_from_ratio(ratio)
    recommendations=[]
    if sum([len(it) for it in models_recommendations]) <= limit:
        # przypadek w którym nie ma wystarczającej liczby artykułów
        for it in models_recommendations:
            recommendations.extend([i for i in it])
        return recommendations 
    else:
        while len(recommendations) < limit:
            p = np.random.rand()
            x = p > vect
            ind = list(x).index(False)
            if len(models_recommendations[ind]) > 0:
                recommendations.append(models_recommendations[ind].pop(0))
            if len(recommendations) != len(set(recommendations)):
                # wyrzucenie powtórek
                recommendations = [it for it in set(recommendations)]
            # print(models_recommendations)
        return recommendations


class Popularity_model:
    MODEL_NAME = "p_model"
    def __init__(self,articles_db,user_db=None,w=(1,1,1)):
        # model przystosowany do testowania Kamila, user w recomm, nie w init
        self.articles = articles_db
        self.user_db=user_db
        self.recommended = []
        self.w = w  # do okreslenia wag

    def get_model_name(self):
        return self.MODEL_NAME

    def head(self,db):
        return db.head()
    
    @staticmethod
    def select_if_no_userdb(art_db,limit):
        '''metoda recomm dla przypadku <user not in database>'''
        selected = art_db.sort_values(by='popularity',ascending=False).head(limit)[['nzz_id']]
        recommended = [item[0] for item in selected.values.tolist()]
        return recommended

    @staticmethod
    def select_if_userdb(art_db, user_db, user, limit,w):
        pass

    def recommend(self, user, articles_to_ignore=[], topn=10, verbose=False):
        '''wyniki systemu rekondacji (lista <limit> wyników)'''
        if self.user_db is None:
            '''przypadek bez zaimplementowanej bazy użytkowników'''
            self.recommended = self.select_if_no_userdb(self.articles, topn)
        elif user not in self.user_db.user_id.values:
            '''przypadek zaimplementowanej bazy użytkowników, użytkownik nie jest w bazie'''
            self.recommended = self.select_if_no_userdb(self.articles, topn)
        else:
            '''przypadek zaimplementowanej bazy użytkownikow, użytkownik jest w bazie'''
            self.recommended = self.select_if_userdb(self.articles, self.user_db, user, topn, w=self.w)
        db = pd.DataFrame(self.recommended,columns=['nzz_id'])
        return db
        

class Popularity_model_merge(Popularity_model):
    '''przypadek zmergowany'''
    # bardzo brzydki, ale tylko do testów x.x
    @staticmethod
    def select_if_userdb(art_db, user_db, user, limit,w):
        rrr, eee = [], []
        user_articles = user_db[user_db['user_id'] == user].iloc[:,1].tolist()
        # popularity
        selected = art_db.sort_values(by='popularity',ascending=False) \
                   .head(limit + len(user_articles))[['nzz_id']].values.tolist()

        rrr.append([item[0] for item in selected][:limit])
        eee.append(1)

        # author
        authors = art_db[art_db['nzz_id'].isin(user_articles)]['author']    # dep. przeczytanych art
        dupl = authors.value_counts()[authors.value_counts()>1].drop(index="Unknown", errors='ignore')

        ratio = tuple(dupl)  # ratio do późniejszego wyboru
        index = list(dupl.index) # index odpowiadający ratio

        if len(ratio) == 0: #brak powtórek
            recommended = []
            ev = 0
        else:
            recomm_for_each = []
            for item in index:
                selected = list(art_db[art_db['author'] == item].sort_values(by='popularity',ascending=False) \
                        .head(limit + len(user_articles))['nzz_id'])
                # dodanie tych, które nie zostaly przeczytane
                recomm_for_each.append([item for item in selected])  
            # wybieram z prawdopodobiństwem (wybrane przeczytane)/(wszystkie przeczytane) artykuły
            recommended = choose_recomm(recomm_for_each,ratio,limit)
            ev = evaluation(ratio)
        rrr.append(recommended)
        eee.append(ev)

        # department
        departs = art_db[art_db['nzz_id'].isin(user_articles)]['department']    # dep. przeczytanych art
        dupl = departs.value_counts()[departs.value_counts()>1].drop(index="Unknown", errors='ignore')
        
        ratio = tuple(dupl)  # ratio do późniejszego wyboru
        index = list(dupl.index) # index odpowiadający ratio
        # print(dupl)
        if len(ratio) == 0: #brak powtarzających się schematów
            recommended = []
            ev = 0
        else:
            recomm_for_each=[]
            for item in index:
                selected = list(art_db[art_db['department'] == item].sort_values(by='popularity',ascending=False) \
                        .head(limit + len(user_articles))['nzz_id'])
                # dodanie tych, które nie zostaly przeczytane
                recomm_for_each.append([item for item in selected])  
            # wybieram z prawdopodobiństwem (wybrane przeczytane)/(wszystkie przeczytane) artykuły
            recommended = choose_recomm(recomm_for_each,ratio,limit)
            ev = evaluation(ratio)

        rrr.append(recommended)
        eee.append(ev)
        rec = choose_recomm(rrr,eee,limit,w)
        return rec


if __name__ == "__main__":
    a = choose_recomm([[1,2,3],[4,5,6],[7,8,9]],(1,1,10),3,w=(100,100,1))
    print(a)