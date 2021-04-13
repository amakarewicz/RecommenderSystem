import pandas as pd


class RandomModel:

    MODEL_NAME = "random_model"

    def __init__(self, random_state):
        self.random_state = random_state

    def get_model_name(self):
        return self.MODEL_NAME

    def fit(self, articles):
        self.articles = articles
        # Ten model nie potrzebuje trenowania

    def recommend(self, topn, verbose=False):
        recommendations_df = self.articles.sample(topn, random_state=self.random_state)

        if not verbose:
            recommendations_df = recommendations_df[["nzz_id"]]

        return recommendations_df
