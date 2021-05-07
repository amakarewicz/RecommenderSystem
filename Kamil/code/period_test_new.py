import pandas as pd
import numpy as np
from typing import Union
from dataclasses import dataclass
from abstract_model_class import Recommendation_model
from cf_model_main import CF_model
from datetime import datetime
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"



def precision(rec: list, user_data: list) -> float:
    """ function giving precision value for list of recommendations and user articles.

    Args:
        rec (list) : articles recommended for user
        user_data (list): articles read by user

    Returns:
        [float, np.nan]: precision value or np.nan if there is no recommendation 
                         or no user articles.
    """
    if len(user_data) != 0 and len(rec) != 0:
        return len([i for i in rec if i in user_data]) / len(rec)
    return np.nan


def recall(rec: list, user_data: list) -> float:
    """ function giving recall value for list of recommendations and user articles.

    Args:
        rec (list) : articles recommended for user
        user_data (list): articles read by user

    Returns:
        [float, np.nan]: recall value or np.nan if there is no recommendation 
                         or no user articles.
    """
    if len(user_data) != 0 and len(rec) != 0:
        return len([i for i in rec if i in user_data]) / len(user_data)
    return np.nan

def f1score(recall: float, precision: float) -> float:
    """ harmonic mean of precision and recall

    Args:
        recall (float): recall value
        precision (float): precision value

    Returns:
        [float, np.nan] : f1_score value, np.nan if recall or precision is np.nan
    """
    if recall == 0 or precision == 0:
        return 0
    elif recall == np.nan or precision == np.nan:
        return np.nan
    return 2/(1/recall + 1/precision)

def get_db(filename: str) -> pd.DataFrame:
    """ function getting pandas DF from .csv file

    Args:
        filename (str): file / filepath

    Returns:
        [pd.DataFrame]: DF from file
    """
    return pd.read_csv(filename)


@dataclass
class period_eval:
    """ test evaluating models precision and recall.
        1. Getting articles read by user from first period.
        2. Creating database containing ALL articles from second period and articles
        read by user from first period
        3. Recommending articles from given database basing ONLY on articles read by user 
        in first period, excluding them later from recommendations (ignore = True).
            > other users have all the interactions - needed for collaborative filthering model
        4. Comparing given recommendations with articles actually read by user from second
        period.

    Args:
        Reverse (bool, optional): if True, swaps train and test articles. Defaults to False
    init atributes defined later.
        articles_1st_period: df containg articles from 1st period
        articles_2nd_period: df containg articles from 2nd period
        readers_1st_period: df containg user interactions from 1st period
        readers_2nd_period: df containg user interactions from 2nd period
    """
    reverse: bool = False

    articles_1st_period: pd.DataFrame = None
    articles_2nd_period: pd.DataFrame = None
    readers_1st_period: pd.DataFrame = None
    readers_2nd_period: pd.DataFrame = None

    @staticmethod
    def user_articles(user_db: pd.DataFrame, user_id: int) -> list:
        """ method returning list of articles read by given user

        Args:
            user_db (pd.DataFrame): list of all users interactions
            user_id (int):  user id

        Returns:
            list: list of articles read by user.
        """
        user_articles = user_db[user_db['user_id'] == user_id].iloc[:,1].tolist()   
        return user_articles

    def readers_in_half(self,art_db: pd.DataFrame, user_db: pd.DataFrame):
        """ method creating articles and user interactions database from each period

        Args:
            art_db (pd.DataFrame): articles dataframe
            user_db (pd.DataFrame): user interactions dataframe
        """
        sorted_art_db = art_db.sort_values(by='pub_date', ascending = not reversed)

        self.articles_1st_period = sorted_art_db[:round(len(art_db)/2)]
        self.articles_2nd_period = sorted_art_db[round(len(art_db)/2):]
        
        self.readers_1st_period = user_db[user_db['nzz_id'].isin(
                                  self.articles_1st_period['nzz_id'])]
        self.readers_2nd_period = user_db[user_db['nzz_id'].isin(
                                  self.articles_2nd_period['nzz_id'])]
        return

    def evaluate_model(self, Model: Recommendation_model, art_db: pd.DataFrame,
                       user_db: pd.DataFrame,limit: list = [5, 10, 15],
                       **kwargs) -> (pd.DataFrame, pd.DataFrame):
        """function evaluating model, returning dataframe of results for each user and
            each number of articles (limit)

        Args:
            Model (Recommendation_model): Model to evaluate
            art_db (pd.DataFrame): articles dataframe
            user_db (pd.DataFrame): user interactions dataframe
            limit (int): number of recommendations for model
            **kwargs: further parameters for model.__init__(),
                      p.e. evaluate_model( Model=Popularity_model_final, ... , w=(100,1,1) )

        Returns:
            [type]: [description]
        """
        # creating subperiods
        self.readers_in_half(art_db,user_db)

        # results of all users
        results_db = []

        # for each user and each number of articles
        for i in range(1,1001):
            # getting articles read by user in first period
            user_articles_1 = self.user_articles(self.readers_1st_period, user_id = i)
            # getting articles read by user in second period
            user_articles_2 = self.user_articles(self.readers_2nd_period, user_id = i)
            # broaden articles from second period by those read by user from 1st period

            articles = self.articles_2nd_period.append(
                    self.articles_1st_period[
                    self.articles_1st_period['nzz_id'].isin(user_articles_1)])

            # BUG: To usuwa artykuły, które przeczytał dany użytkownik u wszystkich użytkowników
            # all the interactions except users from second period
            #interactions = user_db[~user_db["nzz_id"].isin(user_articles_2).dropna()]

            interactions = user_db[~((user_db["user_id"] == i) & (user_db["nzz_id"].isin(user_articles_2)))]

            #gettin recommendations
            model = Model(articles_db=articles, user_db=interactions, **kwargs)

            for l in limit:
                recommended = model.recommend(user_id=i, limit=l,ignored=True)

                # append recall and precision
                pre, rec = precision(recommended,user_articles_2), recall(recommended,user_articles_2)
                results_db.append([ model.get_name(),i,l,len(user_articles_1),
                                    len(user_articles_2), pre, rec, f1score(pre,rec)
                                     ])
        db = pd.DataFrame(results_db, columns=['model','user','number_of_recomm','train_articles',
                                               'test_articles','precision','recall', 'f1score'])
        # mean() for each number of recommendations, for test art, train art > 2
        short_results = db[(db['test_articles'] > 2) & (db['train_articles'] > 2)] \
                        .groupby('number_of_recomm')[['precision','recall', 'f1score']] \
                        .mean().reset_index()

        return short_results, db


if __name__ == "__main__":
    """ example:
    model: Popularity_model
    limit: 5, 10, 15
    """
    

    art_db = get_db("art_clean_wt_all_popularity.csv")
    # shortened db
    art_db = art_db[['nzz_id', 'author', 'department', 'pub_date', 'popularity']]

    user_db = get_db("readers.csv")

    x = period_eval(reverse=False)
    s, r = x.evaluate_model( CF_model,
                             art_db, user_db, limit = [5, 10, 15] )
    print(s)
    # print(r.head())