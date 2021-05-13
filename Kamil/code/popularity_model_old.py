from some_functions import get_db, choose_recomm, evaluation
from abstract_model_class import Recommendation_model
import pandas as pd
from typing import Union

class Popularity_model(Recommendation_model):
    """ Popularity Recommendation Model Class

    defined in Recommendation_model abstract class:

    >> method:  __init__(self, articles_db, user_db)

    Args:
        articles_db (pd.DataFrame, optional): database of articles, containing for each:
                        [nzz_id, author, catchline, content, content_length,
                        department, lead_text, pub_date, title, popularity].
                        Defaults to None.
        user_db (pd.DataFrame, optional): database of users and their read articles, containg:
                        [user_id, nzz_id].
                        Defaults to None.

    >> method: get_name(self)
    method returning self.MODEL_NAME

    Returns:
        str: model name

    >> static method: user_articles(user_db: pd.DataFrame, user_id: int) -> list:
    method returning articles read by given user

    Args:
        user_db (pd.DataFrame): list of all users interactions
        user_id (int):  user id

    Returns:
        list: list of articles read by user.
    """

    MODEL_NAME = "popularity"
    
    def recommend(self, user_id: int, limit: int = 5,ignored: Union[list, bool] = True,
                  ev_return: bool = False, articles_db: pd.DataFrame = None,
                  user_db: pd.DataFrame = None) -> list:
        """ recommend method, returning list of <limit> ID's recommended by model

        Args:
            user_id (int): user id used to find their articles in user_db
            limit (int, optional): number of articles to recommend. Defaults to 5.
            ignored (Union[list, bool], optional): if ignored
                                                   True (default) -> articles read by user
                                                   list -> list of ignored articles
                                                   empty list / False -> nothing ignored
                                                   Defaults to True.
            ev_return (bool, optional): if True, second return is eval value. Defaults to True.
            articles_db (pd.DataFrame, optional): database of articles. Defaults to self...
            user_db (pd.DataFrame, optional): database of users and their read articles self...
        Returns:
          list: list of recommended articles.
          int/ list, optional: evaluation of results / evaluation for each result.
        """
        # allows to give art_db and user_db from outter space, defaults are from object
        if articles_db is None: 
            articles_db = self.articles_db
        if user_db is None:
            user_db = self.user_db

        if user_db is None:
            # user database is not given
            recommended, ev = self._select_if_no_userdb(articles_db, limit, ignored)
        elif user_id not in user_db.user_id.values:
            # user not in user database
            recommended, ev = self._select_if_no_userdb(articles_db, limit, ignored)
        else:
            # user in user database
            recommended, ev = self._select_if_userdb(articles_db, user_db, user_id,
                                                     limit, ignored)
        if ev_return == True:
            return recommended, ev
        return recommended    

    @staticmethod
    def _select_if_no_userdb(art_db: pd.DataFrame, limit: int,
                             ignored: Union[list,bool] ) -> [list, int]:
        """ recommend method for case <user not in DB> and <DB is None> 

        Args:
            art_db (pd.DataFrame): database of articles
            limit (int): database of users and their read articles
            ignored (Union[list,bool]): if ignored:
                                        True (default) -> articles read by user
                                        list -> list of ignored articles
                                        empty list / False -> nothing ignored

        Returns:
            list: list of recommended articles.
            int: evaluation of results.
        """
        # old tactic - just <limit> of the most popular
        # recommended = list(art_db.sort_values(by='popularity',ascending=False)\
        #               .head(limit)['nzz_id'])
        
        # new tactic - the most popular grouped by department, choosing by probability
        selected = art_db.groupby('department')['department_popularity'].min().reset_index() \
                         .sort_values(by='department_popularity', ascending = False)[:-1]
        ratio = tuple(selected['department_popularity'])
        index = list(selected['department'])

        recomm_for_each = []
        if ignored in [True, False, []]:
            # without ignoring any article (or given 'True' to ignore
            # user articles but user is unknown here)
    
            for item in index:
                selected = list(art_db[art_db['department'] == item] \
                           .sort_values(by='popularity',ascending=False).head(limit)['nzz_id'])
                recomm_for_each.append(selected)
        else:
            for item in index:
                selected = list(art_db[art_db['department'] == item] \
                           .sort_values(by='popularity',ascending=False) \
                           .head(limit + len(ignored))['nzz_id'])
                # excluding ignored
                recomm_for_each.append([item for item in selected if item not in ignored]) 
        
        # choosing by probability from ratio p.e. P((1, 2, 3)) = (1/6, 2/6, 3/6)
        recommended = choose_recomm(recomm_for_each,ratio,limit)
        
        return recommended, 1

    @staticmethod
    def _select_if_userdb(art_db: pd.DataFrame, user_db: pd.DataFrame, user_id: int,
                          limit: int, ignored: Union[list,bool]) -> [list, int]:
        """ recommend method for case <user in database>

        Args:
            art_db (pd.DataFrame): database of articles
            user_db (pd.DataFrame): database of users and their read articles
            user_id (int): user id used to find their articles in user_db
            limit (int): number of articles to recommend.
            ignored (Union[list,bool]): if ignored:
                                        True (default) -> articles read by user
                                        list -> list of ignored articles
                                        empty list / False -> nothing ignored
                                                  
        Returns:
            list: list of recommended articles.
            int: evaluation of results.
        """
        if ignored in [False, []]:
            # without ignoring any article
            recommended = list(art_db.sort_values(by='popularity',ascending=False) \
                          .head(limit)['nzz_id'])
            return recommended, 1
        elif ignored is True:
            # ignoring articles read by user
            ignored = Popularity_model.user_articles(user_db, user_id)
        # else: ignoring given articles
        selected = list(art_db.sort_values(by='popularity',ascending=False) \
                .head(limit + len(ignored))['nzz_id'])
        # excluding ignored
        recommended = [item for item in selected if item not in ignored][:limit]
        
        return recommended, 1

    @staticmethod
    def _key_select(name: str, art_db: pd.DataFrame, user_articles: list,
                    limit: int, ignored: Union[list,bool]) -> [list, int]:
        """ selecting articles based on <name> ('department' or 'author')

        Args:
            name (str): pd.groupby column
            art_db (pd.DataFrame): database of articles
            user_articles (list): list of articles read by user.
            limit (int): number of articles to recommend.
            ignored (Union[list,bool]): if ignored:
                                        True (default) -> articles read by user
                                        list -> list of ignored articles
                                        empty list / False -> nothing ignored

        Returns:
            list: list of recommended articles.
            int: evaluation of results.
        """
        # selecting articles
        selected = art_db[art_db['nzz_id'].isin(user_articles)][name]
        # selecting those which <name> occures at least 2 times
        # excluding "Unknown" (no data given)
        dupl = selected.value_counts()[selected.value_counts()>1] \
                       .drop(index="Unknown", errors='ignore')

        # number of occurences
        ratio = tuple(dupl) 
        # list of different <name> values
        index = list(dupl.index)

        if len(ratio) == 0:
            # case when there is no recommendation
            return [], 0

        recomm_for_each = []
        if ignored in [False, []]:
            # without ignoring any article
            for item in index:
                selected = list(art_db[art_db[name] == item] \
                           .sort_values(by='popularity',ascending=False) \
                           .head(limit)['nzz_id'])
                recomm_for_each.append(selected)

        else:
            if ignored is True:
                # ignoring articles read by user
                ignored = user_articles

            # else: ignoring given articles
            for item in index:
                selected = list(art_db[art_db[name] == item] \
                           .sort_values(by='popularity',ascending=False) \
                           .head(limit + len(ignored))['nzz_id'])
                # excluding ignored
                recomm_for_each.append([item for item in selected if item not in ignored]) 
        
        # choosing by probability from ratio p.e. P((1, 2, 3)) = (1/6, 2/6, 3/6)
        recommended = choose_recomm(recomm_for_each,ratio,limit)
        ev = evaluation(ratio)
        return recommended, ev


class Popularity_model_author(Popularity_model):
    """ popularity model with 'author' _key_select """
    
    MODEL_NAME = "author"

    @staticmethod
    def _select_if_userdb(art_db: pd.DataFrame, user_db: pd.DataFrame, user_id: int,
                          limit: int, ignored: Union[list,bool]) -> [list, int]:
        """ recommend method for case <user in database>

        Args:
            art_db (pd.DataFrame): database of articles
            user_db (pd.DataFrame): database of users and their read articles
            user_id (int): user id used to find their articles in user_db
            limit (int): number of articles to recommend.
            ignored (Union[list,bool]): if ignored:
                                        True (default) -> articles read by user
                                        list -> list of ignored articles
                                        empty list / False -> nothing ignored
                                                  
        Returns:
            list: list of recommended articles.
            int: evaluation of results.
        """
        user_articles = Popularity_model.user_articles(user_db, user_id)
        recommended, ev = Popularity_model._key_select(name='author', art_db=art_db,
                                                      user_articles=user_articles,
                                                      limit=limit, ignored=ignored)
        return recommended, ev


class Popularity_model_department(Popularity_model):
    """ popularity model with 'department' _key_select """
    MODEL_NAME = "department"
    
    @staticmethod
    def _select_if_userdb(art_db: pd.DataFrame, user_db: pd.DataFrame, user_id: int,
                          limit: int, ignored: Union[list,bool]) -> [list, int]:
        """ recommend method for case <user in database>

        Args:
            art_db (pd.DataFrame): database of articles
            user_db (pd.DataFrame): database of users and their read articles
            user_id (int): user id used to find their articles in user_db
            limit (int): number of articles to recommend.
            ignored (Union[list,bool]): if ignored:
                                        True (default) -> articles read by user
                                        list -> list of ignored articles
                                        empty list / False -> nothing ignored
                                                  
        Returns:
            list: list of recommended articles.
            int: evaluation of results.
        """
        user_articles = Popularity_model.user_articles(user_db, user_id)
        recommended, ev = Popularity_model._key_select(name='department', art_db=art_db,
                                                      user_articles=user_articles,
                                                      limit=limit, ignored=ignored)
        return recommended, ev


class Popularity_model_final(Popularity_model):
    """ final model selecting by probability from:
        popularity_model.recommend() results
        popularity_model_author.recommend() results
        popularity_model_department.recommend() results
    """
    MODEL_NAME = "final"

    def __init__(self, articles_db: pd.DataFrame, user_db: pd.DataFrame, w: tuple = (19,33,0)):
        """
        Args:
            w (tuple): weight for each of submodels -> (popularity, author, department)
        """
        super().__init__(articles_db,user_db)
        self.w = w

    def _select_if_userdb(self, art_db: pd.DataFrame, user_db: pd.DataFrame, user_id: int,
                          limit: int, ignored: Union[list,bool]) -> [list, tuple]:
        """ recommend method for case <user in database>

        Args:
            art_db (pd.DataFrame): database of articles
            user_db (pd.DataFrame): database of users and their read articles
            user_id (int): user id used to find their articles in user_db
            limit (int): number of articles to recommend.
            ignored (Union[list,bool]): if ignored:
                                        True (default) -> articles read by user
                                        list -> list of ignored articles
                                        empty list / False -> nothing ignored
                                                  
        Returns:
            list: list of recommended articles.
            int: evaluation of results.
        """
        user_articles = Popularity_model.user_articles(user_db, user_id)
        P, Pe = Popularity_model._select_if_userdb(art_db, user_db, user_id, limit, ignored)
        A, Ae = Popularity_model._key_select(name='author', art_db=art_db,
                                                      user_articles=user_articles,
                                                      limit=limit, ignored=ignored)
        D, De = Popularity_model._key_select(name='department', art_db=art_db,
                                                      user_articles=user_articles,
                                                      limit=limit, ignored=ignored)
        # choosing by probability from ratio p.e. P((1, 2, 3)) = (1/6, 2/6, 3/6)
        recommended = choose_recomm([P,A,D],(Pe,Ae,De),limit,w = self.w)

        # score dla każdego z artykułów:
        ev = []
        for item in recommended:
            score = art_db[art_db['nzz_id']==item][['popularity']]
            ev.append(score.values[0][0]**(1/5))
        return recommended, ev
