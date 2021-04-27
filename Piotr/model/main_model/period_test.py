import pandas as pd
import numpy as np
from dataclasses import dataclass
from abstract_model_class import Recommendation_model
import popularity_model # to example

def precision(rec,user_data):
    # recommended and user articles
    if len(user_data) != 0 and len(rec) != 0:
        return len([i for i in rec if i in user_data]) / len(rec)
    return np.nan


def recall(rec,user_data):
    # recommended and user articles
    if len(user_data) != 0 and len(rec) != 0:
        return len([i for i in rec if i in user_data]) / len(user_data)
    return np.nan

def get_db(filename):
    return pd.read_csv(filename)


@dataclass
class period_eval:
    """ test evaluating models precision and recall.
    1. Getting articles read by user from first period.
    2. Creating database containing ALL articles from second period and articles
       read by user from first period
    3. Recommending articles from given database basing ONLY on articles read in first
       period, excluding them from recommendations
    4. Comparing given recommendations with articles actually read by user from second
       period.

    init atributes defined later.
    articles_1st_period: df containg articles from 1st period
    articles_2nd_period: df containg articles from 2nd period
    readers_1st_period: df containg user interactions from 1st period
    readers_2nd_period: df containg user interactions from 2nd period
    """
    articles_1st_period: pd.DataFrame = None
    articles_2nd_period: pd.DataFrame = None
    readers_1st_period: pd.DataFrame = None
    readers_2nd_period: pd.DataFrame = None

    @staticmethod
    def user_articles(user_db: pd.DataFrame, user_id: int) -> list:
        """method returning list of articles read by given user"""
        user_articles = user_db[user_db['user_id'] == user_id].iloc[:,1].tolist()   
        return user_articles

    def readers_in_half(self,art_db: pd.DataFrame, user_db: pd.DataFrame):
        """ method creating articles and user interactions database from each period"""
        self.articles_1st_period = art_db.sort_values(by='pub_date')[:round(len(art_db)/2)]
        self.articles_2nd_period = art_db.sort_values(by='pub_date')[round(len(art_db)/2):]
        
        self.readers_1st_period = user_db[user_db['nzz_id'].isin(
                                  self.articles_1st_period['nzz_id'])]
        self.readers_2nd_period = user_db[user_db['nzz_id'].isin(
                                  self.articles_2nd_period['nzz_id'])]
        return

    def evaluate_model(self, Model: Recommendation_model, art_db: pd.DataFrame,
                       user_db: pd.DataFrame ,limit: list = [5, 10, 15]) -> pd.DataFrame:
        """ function evaluating model, returning dataframe of results for each user and
            each number of articles (limit)
        """
        # creating subperiods
        self.readers_in_half(art_db,user_db)

        # results of all users
        results_db = []

        # for each user and each number of articles
        for i in range(1,1001):
            for l in limit:
                # getting articles read by user in first period
                user_articles_1 = self.user_articles(self.readers_1st_period, user_id = i)
                # broaden articles from second period by thos read by user from 1st period
                articles = self.articles_2nd_period.append(
                        self.articles_1st_period[
                        self.articles_1st_period['nzz_id'].isin(user_articles_1)])

                # gettin recommendations
                model = Model(articles_db=articles,
                            user_db=self.readers_1st_period)
                recommended = model.recommend(user_id=i, limit=l,ignored=True)

                # getting articles read by user in second period
                user_articles_2 = self.user_articles(self.readers_2nd_period, user_id = i)

                # append recall and precision
                results_db.append([ model.get_name(),i,l,len(user_articles_1),
                                    len(user_articles_2), precision(recommended,user_articles_2),
                                    recall(recommended,user_articles_2) ])
        db = pd.DataFrame(results_db, columns=['model','user','number_of_recomm','train_articles','test_articles','precision','recall'])
        
        return db


if __name__ == "__main__":
    """ example:
    model: Popularity_model
    limit: 5, 10, 15
    """
    

    art_db = get_db(r'C:\Users\a814811\OneDrive - Atos\RecommenderSystem\art_clean_wt_all_popularity.csv')
    # shortened db
    art_db = art_db[['nzz_id', 'author', 'department', 'pub_date', 'popularity']]

    user_db = get_db(r'C:\Users\a814811\OneDrive - Atos\RecommenderSystem\readers.csv')

    x = period_eval()
    r = x.evaluate_model( popularity_model.Popularity_model_final,
                          art_db, user_db, limit = [5, 10, 15] )
    print(r)
    print(r.groupby('number_of_recomm')[['precision','recall']].describe())