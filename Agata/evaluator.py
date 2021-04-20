import pandas as pd
import random

class ModelEvaluator:

    # topn accuracy metrics consts
    EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

    def get_read_articles(self,person_id, interactions):
        # Get the user's data and merge in the article info
        interacted_items = interactions.loc[person_id, 'art_id']
        return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

    def get_not_interacted_items_sample(self, person_id, articles, interactions, sample_size, seed=123):
        interacted_items = self.get_read_articles(person_id, interactions)
        all_items = set(articles['nzz_id'])
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

    def evaluate_model_for_user(self, model, person_id, articles, interactions, interactions_train, interactions_test):
        #Getting the items in test set
        interacted_values_testset = interactions_test.loc[person_id]
        if type(interacted_values_testset['art_id']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['art_id'])
        else:
            person_interacted_items_testset = set([(interacted_values_testset['art_id'])])  
        interacted_items_count_testset = len(person_interacted_items_testset) 

        #Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(person_id, 
                                               items_to_ignore=self.get_read_articles(person_id, 
                                                                                    interactions_train), 
                                               topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        #For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            #Getting a random sample (100) items the user has not interacted 
            #(to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id, articles, interactions,
                                                                          sample_size=self.EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, 
                                                                          seed=123)

            #Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            #Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['art_id'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['art_id'].values
            #Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, _ = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, _ = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        #Recall is the rate of the interacted items that are ranked among the Top-N recommended items, 
        #when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_metrics, person_recs_df

    def evaluate_model(self, model, articles, readers, interactions_train, interactions_test):
        # indexing to fasten the search
        interactions_total_ind = readers.set_index('id')
        interactions_train_ind = interactions_train.set_index('id')
        interactions_test_ind = interactions_test.set_index('id')
        #print('Running evaluation for users')
        people_metrics = []
        people_recs = []
        for idx, person_id in enumerate(list(interactions_test_ind.index.unique().values)):
            #if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics, person_recs_df = self.evaluate_model_for_user(model, person_id, articles, interactions_total_ind, interactions_train_ind, interactions_test_ind)  
            person_metrics['_person_id'] = person_id
            person_recs_df['_person_id'] = person_id
            people_metrics.append(person_metrics)
            people_recs.append(person_recs_df)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics).sort_values('interacted_count', ascending=False)
        
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}    
        return global_metrics, detailed_results_df, people_recs
