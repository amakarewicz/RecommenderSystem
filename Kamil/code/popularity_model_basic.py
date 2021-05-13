import pandas as pd


class PopularityModel:

    MODEL_NAME = "popularity_model"

    def __init__(self):
        pass

    def get_model_name(self):
        return self.MODEL_NAME

    def fit(self, readers):
        self.readers = readers
        # Ten model nie potrzebuje trenowania

    def recommend(self, user_id=None, articles_to_ignore=[], topn=10, verbose=False):
        article_popularity = self.readers["nzz_id"].value_counts(sort=True).rename_axis("nzz_id").reset_index(name="read_count")

        recommendations_df = article_popularity[~article_popularity["nzz_id"].isin([])].head(topn)


        if not verbose:
            recommendations_df = recommendations_df[["nzz_id"]]

        return recommendations_df
