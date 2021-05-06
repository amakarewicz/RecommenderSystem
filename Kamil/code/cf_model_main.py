from abstract_model_class import Recommendation_model
from scipy.sparse import csr_matrix
import pandas as pd
import implicit


class CF_model(Recommendation_model):
    """Collaborative Filtering Recommendation Model Class

    defined in Recommendation_model abstract class:

    >> method:  __init__(self, articles_db, user_db, n_latent_factors, regularization, alpha, iterations)

    Args:
        articles_db (pd.DataFrame, optional): database of articles, containing for each:
                        [nzz_id, author, catchline, content, content_length,
                        department, lead_text, pub_date, title, popularity].
                        Defaults to None.
        user_db (pd.DataFrame, optional): database of users and their read articles, containg:
                        [user_id, nzz_id].
                        Defaults to None.
        n_latent_factors (int, optional): number of user/article latent factors to create during matrix factorization.
                        Defaults to 150.0.
        regularization (float, optional): regularization to apply.
                        Defaults to 100.0.
        alpha (float, optional): alpha value to apply in the interaction matrix (represents interaction confidence).
                        Defaults to 50.0.
        iterations (int, optional): number of iterations of matrix approximation to perform.
                        Defaults to 15.

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

    MODEL_NAME = "collaborative filtering"

    def __init__(
        self,
        articles_db=None,
        user_db=None,
        n_latent_factors=150,
        regularization=100.0,
        alpha=50.0,
        iterations=15
    ):
        model = implicit.als.AlternatingLeastSquares(
            factors=n_latent_factors,
            regularization=regularization,
            iterations=iterations,
        )

        reader_article_matrix_df = pd.crosstab(
            user_db["user_id"], user_db["nzz_id"]
        ).fillna(0)

        self.article_ids = {
            k: v for k, v in enumerate(reader_article_matrix_df.columns)
        }
        self.reader_ids = {v: k for k, v in enumerate(reader_article_matrix_df.index)}

        reader_article_matrix = reader_article_matrix_df.to_numpy()
        # Type cast do float bo inczej metoda nie obsługuje
        reader_article_csr_matrix = csr_matrix(reader_article_matrix).asfptype() * alpha

        model.fit(reader_article_csr_matrix.T, show_progress=True)
        self.reader_article_csr_matrix = reader_article_csr_matrix
        self.model = model

        super().__init__(articles_db=articles_db, user_db=user_db)

    def recommend(self, user_id, ignored=True, limit=5, ev_return=False):
        """ recommend method, returning list of <limit> ID's recommended by model

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
        if ignored == True:
            filter_already_liked_items = True
            articles_to_ignore = []
        elif not ignored:
            filter_already_liked_items = False
            articles_to_ignore = []
        elif len(ignored) > 0:
            filter_already_liked_items = False
            articles_to_ignore = [
                k for k, v in self.article_ids.items() if v in ignored
            ]
        if user_id in self.reader_ids:
            sorted_user_predictions = self.model.recommend(
                self.reader_ids[user_id],
                self.reader_article_csr_matrix,
                N=limit,
                filter_already_liked_items=filter_already_liked_items,
                filter_items=articles_to_ignore,
            )
            recommendations_list = [
                self.article_ids[prediction[0]]
                for prediction in sorted_user_predictions
            ]
        else:
            recommendations_list = []

        if ev_return:
            scores = [
                prediction[1]
                for prediction in sorted_user_predictions
            ]
            return recommendations_list, scores
        else:
            return recommendations_list
