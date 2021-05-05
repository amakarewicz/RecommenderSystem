from some_functions import get_db, choose_recomm, evaluation
from abstract_model_class import Recommendation_model
import pandas as pd
from typing import Union




class Recommendation(Recommendation_model):
    """ final model selecting by probability from:
        popularity_model_final.recommend() results
        cf_model.recommend() results
        content_based_model.recommend() results
    """
    
    def __init__(self, articles_db: pd.DataFrame, user_db: pd.DataFrame, w: tuple = (1,1,1)):
        """
        Args:
            articles_db (pd.DataFrame, optional): database of articles, containing for each:
                       [nzz_id, author, catchline, content, content_length,
                       department, lead_text, pub_date, title, popularity].
                       Defaults to None.
            user_db (pd.DataFrame, optional): database of users and their read articles, containg:
                       [user_id, nzz_id].
                       Defaults to None.
            w (tuple): weight for each of submodels: 
                       popularity, colaborative filtering, content based.
                       Defaults to None.
        """
        super().__init__(articles_db,user_db)
        self.w = w

    def recommend(self, user_id: int, limit: int = 5,ignored: Union[list, bool] = True,
                  ev_return: bool = False) -> list:
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

        Returns:
          list: list of recommended articles.
          int, optional: evaluation of results.
        """
        # na razie niezależnie od liczby rekomendacji, wagi zniesiemy po testach

        recomm_for_each = []
        for model in [popularity_model_final,
                     cf_model,
                     content_based_model]:
            recomm_for_each.append(model.recommend(user_id, limit=limit, ignored=ignored,
                                   articles_db=self.articles_db, user_db=self.user_db))
        # choose using probability from given weights as ratio tuple, p.e. (1,2,3)
        recommended = choose_recomm(recomm_for_each, self.w)
        return recommended

if __name__ == "__main__":
    art_db = get_db(r'C:\Users\a814811\OneDrive - Atos\RecommenderSystem\art_clean_wt_all_popularity.csv')
    user_db = get_db(r'C:\Users\a814811\OneDrive - Atos\RecommenderSystem\readers.csv')
    a = Recommendation(art_db, user_db)
