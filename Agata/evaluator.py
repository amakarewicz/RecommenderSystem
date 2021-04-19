import pandas as pd
import random

# do poprawy !!!
class ModelEvaluator:
    # Top-N accuracy metrics consts
    EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

    def _get_items_interacted(self, person_id, interactions_df):
        # Get the user's data and merge in the movie information.
        interacted_items = interactions_df.loc[person_id]["nzz_id"]
        return set(
            interacted_items
            if type(interacted_items) == pd.Series
            else [interacted_items]
        )

    def _get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        interacted_items = self._get_items_interacted(
            person_id, self.interactions_full_indexed_df
        )
        all_items = set(self.readers["nzz_id"])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):
        try:
            index = next(i for i, c in enumerate(recommended_items) if c == item_id)
        except:
            index = -1
        hit = int(index in range(0, topn))
        return hit, index

    def _evaluate_model_for_user(self, model, person_id):
        # Getting the items in test set
        interacted_values_testset = self.interactions_test_indexed_df.loc[person_id]
        if type(interacted_values_testset["nzz_id"]) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset["nzz_id"])
        else:
            person_interacted_items_testset = set([interacted_values_testset["nzz_id"]])
        interacted_items_count_testset = len(person_interacted_items_testset)

        # Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend(
            person_id,
            articles_to_ignore=self._get_items_interacted(
                person_id, self.interactions_train_indexed_df
            ),
            topn=10000000000,
        )

        hits_at_5_count = 0
        hits_at_10_count = 0
        # For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            # Getting a random sample (100) items the user has not interacted
            # (to represent items that are assumed to be no relevant to the user)
            # seed = int.from_bytes(item_id.encode('utf-8'), 'little')
            non_interacted_items_sample = self._get_not_interacted_items_sample(
                person_id,
                sample_size=self.EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS,
                seed=12,
            )

            # Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[
                person_recs_df["nzz_id"].isin(items_to_filter_recs)
            ]
            valid_recs = valid_recs_df["nzz_id"].values
            # Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        # Recall is the rate of the interacted items that are ranked among the Top-N recommended items,
        # when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {
            "hits@5_count": hits_at_5_count,
            "hits@10_count": hits_at_10_count,
            "interacted_count": interacted_items_count_testset,
            "recall@5": recall_at_5,
            "recall@10": recall_at_10,
        }
        return person_metrics

    def evaluate_model(self, model, readers, readers_train, readers_test):
        # Indexing by personId to speed up the searches during evaluation
        self.readers = readers
        self.interactions_full_indexed_df = readers.set_index("user_id")
        self.interactions_train_indexed_df = readers_train.set_index("user_id")
        self.interactions_test_indexed_df = readers_test.set_index("user_id")

        # print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(
            list(self.interactions_test_indexed_df.index.unique().values)
        ):
            # if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self._evaluate_model_for_user(model, person_id)
            person_metrics["_person_id"] = person_id
            people_metrics.append(person_metrics)
        print("%d users processed" % idx)

        detailed_results_df = pd.DataFrame(people_metrics).sort_values(
            "interacted_count", ascending=False
        )

        global_recall_at_5 = detailed_results_df["hits@5_count"].sum() / float(
            detailed_results_df["interacted_count"].sum()
        )
        global_recall_at_10 = detailed_results_df["hits@10_count"].sum() / float(
            detailed_results_df["interacted_count"].sum()
        )

        global_metrics = {
            "modelName": model.get_model_name(),
            "recall@5": global_recall_at_5,
            "recall@10": global_recall_at_10,
        }
        return global_metrics, detailed_results_df


model_evaluator = ModelEvaluator()
