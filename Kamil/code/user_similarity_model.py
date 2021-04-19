import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity


class UserSimilarityModel:

    MODEL_NAME = "user_similarity_model"

    def __init__(self, n_latent_factors, random_state=None):
        self.n_latent_factors = n_latent_factors
        self.random_state = random_state

    def get_model_name(self):
        return self.MODEL_NAME

    def fit(self, readers, articles=None):

        reader_article_matrix_df = pd.crosstab(
            readers["user_id"], readers["nzz_id"]
        ).fillna(0)
        reader_ids = list(reader_article_matrix_df.index)
        self.reader_ids = reader_ids

        reader_article_matrix = reader_article_matrix_df.to_numpy()
        # Type cast do float bo inczej metoda nie obsługuje
        reader_article_csr_matrix = csr_matrix(reader_article_matrix).asfptype()

        U, sigma, Vt = svds(reader_article_csr_matrix, k=self.n_latent_factors)
        sigma = np.diag(sigma)

        cosine = cosine_similarity(U)
        np.fill_diagonal(cosine, 0)
        latent_factor_similarity = pd.DataFrame(
            cosine, index=reader_article_matrix_df.index
        )
        latent_factor_similarity.columns = reader_article_matrix_df.index
        self.latent_factor_similarity = latent_factor_similarity

        self.readers = readers
        self.articles = articles
        # Ten model nie potrzebuje trenowania

    def recommend(self, user_id, articles_to_ignore=[], topn=5, verbose=False):
        #TODO: Add multiple N user recoomendation (more users viewed one item then the recommendation has higher confidence)
        top_similar_user_id = self.latent_factor_similarity.iloc[user_id-1].idxmax()
        print(top_similar_user_id)
        sorted_user_predictions = (
            self.readers[self.readers["user_id"] == top_similar_user_id]
            # .sort_values(ascending=False)
            # .reset_index()
            # .rename(columns={user_id: "recommendation_strength"})
        )

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = (
            sorted_user_predictions[
                ~sorted_user_predictions["nzz_id"].isin(articles_to_ignore)
            ]
            # .sort_values("recommendation_strength", ascending=False)
        ).sample(random_state=self.random_state).head(topn)

        if not verbose:
            #TODO: Implement verbose mode from cf_model
            recommendations_df = recommendations_df[["nzz_id"]]

        return recommendations_df
