import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit


class ImplicitModel:

    MODEL_NAME = "implicit_model"

    def __init__(self, n_latent_factors, regularization=0.0, alpha=1.0, iterations=15):
        self.alpha = alpha
        self.model = implicit.als.AlternatingLeastSquares(
            factors=n_latent_factors,
            regularization=regularization,
            iterations=iterations,
        )

    def get_model_name(self):
        return self.MODEL_NAME

    def fit(self, users, articles=None):
        reader_article_matrix_df = pd.crosstab(
            users["user_id"], users["nzz_id"]
        ).fillna(0)
        self.article_ids = {k: v for k, v in enumerate(reader_article_matrix_df.columns)}
        self.reader_ids = {v: k for k, v in enumerate(reader_article_matrix_df.index)}

        reader_article_matrix = reader_article_matrix_df.to_numpy()
        # Type cast do float bo inczej metoda nie obsługuje
        reader_article_csr_matrix = csr_matrix(reader_article_matrix).asfptype() * self.alpha

        self.model.fit(reader_article_csr_matrix.T, show_progress=False)
        # U, sigma, Vt = svds(reader_article_csr_matrix, k=self.n_latent_factors)
        # sigma = np.diag(sigma)
        #
        # reader_predictions = np.dot(np.dot(U, sigma), Vt)
        # reader_predictions_norm = (reader_predictions - reader_predictions.min()) / (
        #    reader_predictions.max() - reader_predictions.min()
        # )
        
        self.reader_article_csr_matrix = reader_article_csr_matrix
        self.articles = articles

    def recommend(self, user_id, articles_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions

        sorted_user_predictions = self.model.recommend(
            self.reader_ids[user_id],
            self.reader_article_csr_matrix,
            N=topn,
            filter_already_liked_items=False,
            filter_items=articles_to_ignore,
        )
        sorted_user_predictions = [(self.article_ids[prediction[0]], prediction[1]) for prediction in sorted_user_predictions]
        sorted_user_predictions_df = pd.DataFrame(sorted_user_predictions, columns=["nzz_id", "recommendation_strength"])

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = (
            sorted_user_predictions_df[
                ~sorted_user_predictions_df["nzz_id"].isin(articles_to_ignore)
            ].sort_values("recommendation_strength", ascending=False)
        ).head(topn)

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
                    "paragraph",
                    "department",
                    "lead_text",
                    "pub_date",
                ]
            ]

        return recommendations_df
