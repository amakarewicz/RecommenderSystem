from re import A
import pandas as pd
import numpy as np
import random
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

class ModelEvaluator:
    # Top-N accuracy metrics consts
    EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100
    
    def __init__(self, k_list):
        self.k_list = k_list

    def dcg_at_k(self, r, k, method=0):
        r= np.asfarray(r)[:k]
        if r.size:
            if method == 0:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size +1)))
            elif method == 1:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError("method must be 0 or 1")
        return 0

    def ndcg_at_k(self, r, k, method=0):
        dcg_max = self.dcg_at_k(sorted(r, reverse=True), k, method)
        if not dcg_max:
            return 0
        return self.dcg_at_k(r, k, method) / dcg_max

    def _get_items_interacted(self, person_id, interactions_df):
        # Get the user's data and merge in the movie information.
        interacted_items = interactions_df.loc[person_id]["nzz_id"]
        return set(
            interacted_items
            if type(interacted_items) == pd.Series
            else [interacted_items]
        )

    def _get_not_interacted_items_sample(self, person_id, sample_size, seed=None):
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

        person_metrics = {}
        for k in self.k_list:
            hits_at_k_count = 0
            ndcg_scores_at_k_for_user = []
        # For each item the user has interacted in test set
            for item_id in person_interacted_items_testset:
                # Getting a random sample (100) items the user has not interacted
                # (to represent items that are assumed to be no relevant to the user)
                # seed = int.from_bytes(item_id.encode('utf-8'), 'little')
                non_interacted_items_sample = self._get_not_interacted_items_sample(
                    person_id,
                    sample_size=self.EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS,
                )

                # Combining the current interacted item with the 100 random items
                items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))
                # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
                valid_recs_df = person_recs_df[
                    person_recs_df["nzz_id"].isin(items_to_filter_recs)
                ]
                valid_recs = valid_recs_df["nzz_id"].values

                # Verifying if the current interacted item is among the Top-N recommended items
                hit_at_k, index_at_k = self._verify_hit_top_n(item_id, valid_recs, k)
                relevance_array_at_k = np.zeros(k)
                if(hit_at_k):
                    relevance_array_at_k[index_at_k] = 1
                
                hits_at_k_count += hit_at_k

                ndcg_scores_at_k_for_user.append(self.ndcg_at_k(relevance_array_at_k, len(relevance_array_at_k), method=1))
            # Recall is the rate of the interacted items that are ranked among the Top-N recommended items,
            # when mixed with a set of non-relevant items
            recall_at_k = hits_at_k_count / float(interacted_items_count_testset)
            precision_at_k = hits_at_k_count / (float(k) * float(interacted_items_count_testset))
            ndcg_at_k_score = sum(ndcg_scores_at_k_for_user) / len(ndcg_scores_at_k_for_user)

            if precision_at_k + recall_at_k == 0:
                # W niektórych przypadkach występuję dzielenie przez 0, dlatego zakładamy wtedy f1-score jako 0
                f1_score_at_k = 0
            else:
                f1_score_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)

            person_metrics[f"hits@{k}_count"] = hits_at_k_count
            person_metrics["interacted_count"] = interacted_items_count_testset
            person_metrics[f"recall@{k}"] = recall_at_k
            person_metrics[f"precision@{k}"] = precision_at_k
            person_metrics[f"f1_score@{k}"] = f1_score_at_k
            person_metrics[f"ndcg@{k}"] = ndcg_at_k_score

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
            #all_recs_at_5.append(person_recs_df["nzz_id"].head(5).values)
            #all_recs_at_10.append(person_recs_df["nzz_id"].head(10).values)
            person_metrics["_person_id"] = person_id
            people_metrics.append(person_metrics)
        print("%d users processed" % idx)

        detailed_results_df = pd.DataFrame(people_metrics).sort_values(
            "interacted_count", ascending=False
        )

 
        global_metrics = {"modelName": model.get_model_name(),}
        for k in self.k_list:
            global_recall_at_k = detailed_results_df[f"recall@{k}"].mean()
            global_precision_at_k = detailed_results_df[f"precision@{k}"].mean()
            global_f1_score_at_k = detailed_results_df[f"f1_score@{k}"].mean()
            global_ndcg_at_k = detailed_results_df[f"ndcg@{k}"].mean()

            global_metrics[f"recall@{k}"] = global_recall_at_k
            global_metrics[f"precision@{k}"] = global_precision_at_k
            global_metrics[f"f1_score@{k}"] = global_f1_score_at_k
            global_metrics[f"ndcg@{k}"] = global_ndcg_at_k

        #coverage_at_5 = self._get_coverage(all_recs_at_5, self.interactions_train_indexed_df["nzz_id"].unique())
        #coverage_at_10 = self._get_coverage(all_recs_at_10, self.interactions_train_indexed_df["nzz_id"].unique())

        return global_metrics, detailed_results_df
