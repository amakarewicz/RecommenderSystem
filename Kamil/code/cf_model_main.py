from numpy import product
from abstract_model_class import Recommendation_model
from scipy.sparse import csr_matrix
import pandas as pd
import implicit

class CF_model(Recommendation_model):

    MODEL_NAME = "collaborative filtering"

    def __init__(self, articles_db=None, user_db=None, n_latent_factors=150, regularization=100.0, alpha=50.0, iterations=15):
        model = implicit.als.AlternatingLeastSquares(
            factors=n_latent_factors,
            regularization=regularization,
            iterations=iterations,
        )

        reader_article_matrix_df = pd.crosstab(
            user_db["user_id"], user_db["nzz_id"]
        ).fillna(0)
        self.article_ids = {k: v for k, v in enumerate(reader_article_matrix_df.columns)}
        self.reader_ids = {v: k for k, v in enumerate(reader_article_matrix_df.index)}

        reader_article_matrix = reader_article_matrix_df.to_numpy()
        # Type cast do float bo inczej metoda nie obsługuje
        reader_article_csr_matrix = csr_matrix(reader_article_matrix).asfptype() * alpha

        model.fit(reader_article_csr_matrix.T, show_progress=False)
        
        self.reader_article_csr_matrix = reader_article_csr_matrix
        self.model = model


        super().__init__(articles_db=articles_db, user_db=user_db)

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
        if user_id in self.reader_ids:
            sorted_user_predictions = self.model.recommend(
                self.reader_ids[user_id],
                self.reader_article_csr_matrix,
                N=limit,
                filter_already_liked_items=filter_already_liked_items,
                filter_items=articles_to_ignore,
            )
            recommendations_list = [self.article_ids[prediction[0]] for prediction in sorted_user_predictions]
        else:
            recommendations_list = []
        print(recommendations_list)
        return recommendations_list
