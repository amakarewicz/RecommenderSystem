import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


class CFModel:

    MODEL_NAME = "CF_model"

    def __init__(self, n_latent_factors):
        self.n_latent_factors = n_latent_factors

    def get_model_name(self):
        return self.MODEL_NAME

    def fit(self, users, articles=None):
        reader_article_matrix_df = pd.crosstab(
            users["user_id"], users["nzz_id"]
        ).fillna(0)
        reader_ids = list(reader_article_matrix_df.index)

        reader_article_matrix = reader_article_matrix_df.to_numpy()
        # Type cast do float bo inczej metoda nie obsługuje
        reader_article_csr_matrix = csr_matrix(reader_article_matrix).asfptype()

        U, sigma, Vt = svds(reader_article_csr_matrix, k=self.n_latent_factors)
        sigma = np.diag(sigma)

        reader_predictions = np.dot(np.dot(U, sigma), Vt)
        reader_predictions_norm = (reader_predictions - reader_predictions.min()) / (
            reader_predictions.max() - reader_predictions.min()
        )

        cf_predictions_df = pd.DataFrame(
            reader_predictions_norm,
            columns=reader_article_matrix_df.columns,
            index=reader_ids,
        ).transpose()

        self.cf_predictions_df = cf_predictions_df
        self.articles = articles

    def recommend(self, user_id, articles_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = (
            self.cf_predictions_df[user_id]
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={user_id: "recommendation_strength"})
        )

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = (
            sorted_user_predictions[
                ~sorted_user_predictions["nzz_id"].isin(articles_to_ignore)
            ]
            .sort_values("recommendation_strength", ascending=False)
            .head(topn)
        )

        if verbose:
            if self.articles is None:
                raise Exception('"articles" are required in verbose mode')

            recommendations_df = recommendations_df.merge(
                self.articles, how="left", left_on="nzz_id", right_on="nzz_id"
            )[
                [
                    "recommendation_strength",
                    "nzz_id",
                    "catchline",
                    "content",
                    "department",
                    "lead_text",
                    "pub_date",
                ]
            ]

        return recommendations_df
