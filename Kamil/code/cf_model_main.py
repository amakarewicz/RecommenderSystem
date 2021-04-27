from numpy import product
from abstract_model_class import Recommendation_model
from scipy.sparse import csr_matrix
import pandas as pd
import implicit

class CF_model(Recommendation_model):

    MODEL_NAME = "collaborative filtering"

    def fit(self, n_latent_factors=150, regularization=100.0, alpha=50.0, iterations=15):
        model = implicit.als.AlternatingLeastSquares(
            factors=n_latent_factors,
            regularization=regularization,
            iterations=iterations,
        )

        reader_article_matrix_df = pd.crosstab(
            self.user_db["user_id"], self.user_db["nzz_id"]
        ).fillna(0)
        self.article_ids = {k: v for k, v in enumerate(reader_article_matrix_df.columns)}
        self.reader_ids = {v: k for k, v in enumerate(reader_article_matrix_df.index)}

        reader_article_matrix = reader_article_matrix_df.to_numpy()
        # Type cast do float bo inczej metoda nie obsługuje
        reader_article_csr_matrix = csr_matrix(reader_article_matrix).asfptype() * alpha

        model.fit(reader_article_csr_matrix.T, show_progress=False)
        
        self.reader_article_csr_matrix = reader_article_csr_matrix
        self.model = model

    def recommend(self, user_id, ignored=True, limit=5):
        # Get and sort the user's predictions

        if ignored == True:
            filter_already_liked_items = True
            articles_to_ignore = []
        elif not ignored:
            filter_already_liked_items = False
            articles_to_ignore = []
        elif len(ignored) > 0:
            filter_already_liked_items = False
            articles_to_ignore = [k for k, v in self.article_ids.items() if v in ignored]

        sorted_user_predictions = self.model.recommend(
            self.reader_ids[user_id],
            self.reader_article_csr_matrix,
            N=limit,
            filter_already_liked_items=filter_already_liked_items,
            filter_items=articles_to_ignore,
        )
        sorted_user_predictions = [(self.article_ids[prediction[0]], prediction[1]) for prediction in sorted_user_predictions]
        sorted_user_predictions_df = pd.DataFrame(sorted_user_predictions, columns=["nzz_id", "recommendation_strength"])

        recommendations_list = sorted_user_predictions_df["nzz_id"].tolist()

        return recommendations_list
