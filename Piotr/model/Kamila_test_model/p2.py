from some_functions import get_db, choose_recomm, evaluation
import pandas as pd
from abstract_model_class import Recommendation_model

class Popularity_model(Recommendation_model):
    '''
    method __init__()
        :param articles_db: database of articles, containing for each:
            [nzz_id, author, catchline, content, content_length,
                department, lead_text, pub_date, title, popularity]
        :type arg: pandas table

        :param user_db: database of users and their read articles, containg:
            [user_id, nzz_id]
        :type arg: pandas table

    method get_name()
        method returning self.MODEL_NAME
    '''

    MODEL_NAME = "popularity"

    def head(self,db):
        '''returns head of chosen database'''
        return db.head()

    @staticmethod
    def user_articles(user_db, user_id):
        '''method returning articles read by given user'''
        user_articles = user_db[user_db['user_id'] == user_id].iloc[:,1].tolist()   
        return user_articles
    
    def recommend(self, user, articles_to_ignore=[], topn=10, verbose=False):
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
        if self.user_db is None:
            '''user database is not given'''
            recommended = self._select_if_no_userdb(self.articles_db, topn)
        elif user not in self.user_db.user_id.values:
            '''user not in user database'''
            recommended = self._select_if_no_userdb(self.articles_db, topn)
        else:
            '''user in user database'''
            recommended = self._select_if_userdb(self.articles_db,
                                                          self.user_db, user,
                                                          topn, articles_to_ignore)
        db = pd.DataFrame(recommended,columns=['nzz_id'])
        return db

    @staticmethod
    def _select_if_no_userdb(art_db,limit):
        '''recommend method for case <user not in DB> and <DB is None>'''
        recommended = list(art_db.sort_values(by='popularity',ascending=False).head(limit)['nzz_id'])
        return recommended

    @staticmethod
    def _select_if_userdb(art_db, user_db, user_id, limit, ignored):
        '''recommend method for case <user in database>'''
        if ignored in [False, []]:
            # without ignoring any article
            recommended = list(art_db.sort_values(by='popularity',ascending=False).head(limit)['nzz_id'])
            return recommended
        elif ignored is True:
            # ignoring articles read by user
            ignored = Popularity_model.user_articles(user_db, user_id)  # artykuły przeczytane
        # else: ignoring given articles
        selected = list(art_db.sort_values(by='popularity',ascending=False) \
                .head(limit + len(ignored))['nzz_id'])
        # excluding ignored
        recommended = [item for item in selected if item not in ignored][:limit]    # wyrzucam powtórki
        
        return recommended

    @staticmethod
    def _key_select(name,art_db,user_articles, limit, ignored):
        '''selecting articles based on <name> ('department' or 'author')'''
        # selecting articles
        selected = art_db[art_db['nzz_id'].isin(user_articles)][name]
        # selecting those which <name> occures at least 2 times
        # excluding "Unknown" (no data given)
        dupl = selected.value_counts()[selected.value_counts()>1].drop(index="Unknown", errors='ignore')

        # number of occurences
        ratio = tuple(dupl) 
        # list of different <name> values
        index = list(dupl.index)

        if len(ratio) == 0:
            # case when there is no recommendation
            return [], 0

        if ignored in [False, []]:
            # without ignoring any article
            recomm_for_each = []
            for item in index:
                selected = list(art_db[art_db[name] == item].sort_values(by='popularity',ascending=False) \
                        .head(limit)['nzz_id'])
                recomm_for_each.append(selected)

        else:
            if ignored is True:
                # ignoring articles read by user
                ignored = user_articles

            # else: ignoring given articles
            recomm_for_each = []
            for item in index:
                selected = list(art_db[art_db[name] == item].sort_values(by='popularity',ascending=False) \
                        .head(limit + len(ignored))['nzz_id'])
                # excluding ignored
                recomm_for_each.append([item for item in selected if item not in ignored]) 
        
        # choosing by probability from ratio p.e. P((1, 2, 3)) = (1/6, 2/6, 3/6)
        recommended = choose_recomm(recomm_for_each,ratio,limit)
        ev = evaluation(ratio)
        return recommended, ev


class Popularity_model_author(Popularity_model):
    '''popularity model with 'author' _key_select'''
    
    MODEL_NAME = "author"

    @staticmethod
    def _select_if_userdb(art_db, user_db, user_id, limit, ignored):
        '''recommend method for case <user in database>'''
        user_articles = Popularity_model.user_articles(user_db, user_id)
        recommended, _ = Popularity_model._key_select(name='author', art_db=art_db,
                                                      user_articles=user_articles,
                                                      limit=limit, ignored=ignored)
        return recommended


class Popularity_model_department(Popularity_model):
    '''popularity model with 'department' _key_select'''
    MODEL_NAME = "department"
    
    @staticmethod
    def _select_if_userdb(art_db, user_db, user_id, limit, ignored):
        '''recommend method for case <user in database>'''
        user_articles = Popularity_model.user_articles(user_db, user_id)
        recommended, _ = Popularity_model._key_select(name='department', art_db=art_db,
                                                      user_articles=user_articles,
                                                      limit=limit, ignored=ignored)
        return recommended


class Popularity_model_final(Popularity_model):
    '''final model selecting by probability from:
        popularity_model.recommend results
        popularity_model_author.recommend results
        popularity_model_department.recommend results
    '''
    MODEL_NAME = "final"

    @staticmethod
    def _select_if_userdb(art_db, user_db, user_id, limit, ignored):
        '''recommend method for case <user in database>'''
        user_articles = Popularity_model.user_articles(user_db, user_id)
        P  = Popularity_model._select_if_userdb(art_db, user_db, user_id, limit, ignored)
        A, Ae = Popularity_model._key_select(name='author', art_db=art_db,
                                                      user_articles=user_articles,
                                                      limit=limit, ignored=ignored)
        D, De = Popularity_model._key_select(name='department', art_db=art_db,
                                                      user_articles=user_articles,
                                                      limit=limit, ignored=ignored)
        # choosing by probability from ratio p.e. P((1, 2, 3)) = (1/6, 2/6, 3/6)
        recommended = choose_recomm([P,A,D],(1,Ae,De),limit)
        return recommended