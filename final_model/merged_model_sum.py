from numpy.lib.utils import source
from abstract_model_class import Recommendation_model
import pandas as pd
import numpy as np
from typing import Union
from some_functions import get_db, evaluation, prob_vector_from_ratio
from popularity_model import Popularity_model_final
from cf_model_main import CF_model

# to z sumowaniem jako wybór, o czym ola mówiła


def choose_recomm(
    models_recommendations: list, w_inner: list, limit: int, ratio: tuple = (1, 1, 1)
) -> list:

    """function chosing for selected recommendations from the models,
        the results from each set according to probability.

    Args:
        models_recommendations (list): list of lists of recommendations
                                       for each model - [[r11,r12,...],[r21,r22,...],...]
        ratio (tuple): default ratio for models
        limit (int): number of recommendations to return
        w_inner (tuple): inner weight for each article: [[w11,w12,...],[w21,w22,...],...]

    Raises:
        ValueError: ratio length is different than number of models.

    Returns:
        list: chosen recommendations according to probability taken from ratio
    """
    if (len(models_recommendations) != len(ratio)) or (len(w_inner) != len(ratio)):
        raise ValueError

    # giving outter weight to inner weight
    for i, evs in enumerate(w_inner):
        w_inner[i] = list(np.array(evs) * ratio[i])

    recommendations = []

    if sum([len(it) for it in models_recommendations]) <= limit:
        # case: no enough articles
        for it in models_recommendations:
            recommendations.extend([i for i in it])
        return recommendations

    else:
        all_evs = []
        for i, item in enumerate(models_recommendations):
            for j, rec in enumerate(item):
                if rec not in recommendations:
                    recommendations.append(rec)
                    all_evs.append(w_inner[i][j])
                else:
                    ind = recommendations.index(rec)
                    all_evs[ind] += w_inner[i][j]

        recommendations = [el for _, el in sorted(zip(all_evs, recommendations), reverse = True)]
        evs =[el for el, _ in sorted(zip(all_evs, recommendations), reverse = True)]
        return recommendations, evs


class MergedModelSum(Recommendation_model):

    """final model selecting by probability from:
    popularity_model_final.recommend() results
    cf_model.recommend() results
    content_based_model.recommend() results

    """

    def __init__(
        self, articles_db: pd.DataFrame, user_db: pd.DataFrame, matrix: np.array = None, feature_names: np.array = None,
        similarity_model: object = None, article_similarity: int = 0.5, keyword_similarity: int = 0.4, w: tuple = (1, 1)):

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

        super().__init__(articles_db, user_db, matrix, feature_names, similarity_model, article_similarity, keyword_similarity)

        self.models = [ Popularity_model_final(articles_db=self.articles_db, user_db=self.user_db),
                        CF_model(articles_db=self.articles_db, user_db=self.user_db) ]
                    
        self.w = w

    def recommend(
        self, user_id: int, limit: int = 5, ignored: Union[list, bool] = True,
        filtering = False) -> list:

        """recommend method, returning list of <limit> ID's recommended by model

        Args:
            user_id (int): user id used to find their articles in user_db
            limit (int, optional): number of articles to recommend. Defaults to 5.
            ignored (Union[list, bool], optional): if ignored
                                                   True (default) -> articles read by user
                                                   list -> list of ignored articles
                                                   empty list / False -> nothing ignored
                                                   Defaults to True.

        Returns:
          list: list of recommended articles.
        """

        # sources = []
        if not ((self.user_db is None) or (user_id not in self.user_db["user_id"].values)):
            # user in user database
            
            recomm_for_each = []
            evaluations = []

            for model in self.models:
                # tutaj zmienic z wagami dla poszczególnego modelu
                recommendation, evaluation = model.recommend(
                    user_id,
                    limit=100,
                    ignored=ignored,
                    articles_db=self.articles_db,
                    user_db=self.user_db,
                    ev_return=True)
                
                # sources.extend([(recomm, model.get_name()) for recomm in recommendation])
                recomm_for_each.append(recommendation) 
                evaluations.append(evaluation)

            # choose using probability from given weights as ratio tuple, p.e. (1,2)
            recommended, _ = choose_recomm(recomm_for_each, evaluations, limit, self.w)

            # filtering out simmilar items
            if filtering:
                recommended = self.filter_out_similar(recommended)

            # limit of articles recommended
            recommended = recommended[:limit]
            
        else:
            # user not in user database or no db
            # recommend from popularity model
            recommended = self.models[0].recommend(
                            user_id,
                            limit=limit,
                            ignored=ignored,
                            articles_db=self.articles_db,
                            user_db=self.user_db)

        # print(f"sources: {sources}")

        return recommended


if __name__ == "__main__":

    art_db = get_db("art_clean_wt_all_popularity.csv")
    user_db = get_db("readers.csv")
    m_merged = MergedModelSum(art_db, user_db, w=(1, 1))
    print(f"merged recommendations:{m_merged.recommend(user_id = 1, limit = 15)}")
