import pandas as pd


class RandomModel:

    MODEL_NAME = "random_model"

    def __init__(self, random_state=None):
        self.random_state = random_state

    def get_model_name(self):
        return self.MODEL_NAME

    def fit(self, articles):
        self.articles = articles
        # Ten model nie potrzebuje trenowania

    def recommend(self, user_id=None, articles_to_ignore=[], topn=10, verbose=False):
        recommendations_df = self.articles[~self.articles["nzz_id"].isin(articles_to_ignore)]
        recommendations_df = recommendations_df.sample(frac=1, random_state=self.random_state).head(topn)

        if not verbose:
            recommendations_df = recommendations_df[["nzz_id"]]

        return recommendations_df
