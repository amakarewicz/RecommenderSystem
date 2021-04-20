
from popularity_model import *
import pandas as pd
import numpy as np
from some_functions import get_db
from sklearn.model_selection import train_test_split
from model_evaluator import ModelEvaluator

# init 
readers = pd.read_csv(r"C:\Users\a814811\OneDrive - Atos\RecommenderSystem\readers.csv")
readers = readers.rename(columns={"id":"user_id", "art_id":"nzz_id"})
art_db = get_db(r'C:\Users\a814811\OneDrive - Atos\RecommenderSystem\art_clean_wt_all_popularity.csv')
art_db = art_db.loc[:,['nzz_id','author','department','popularity']] #skrócenie do potrzebnych rzeczy

read_counts = readers["user_id"].value_counts(sort=True)
read_counts = read_counts.rename_axis("user_id").reset_index(name="read_count")
read_counts = read_counts[read_counts["read_count"] > 3]
readers = readers[readers["user_id"].isin(read_counts["user_id"])]

random_state = None
readers_train, readers_test = train_test_split(readers,
                                   stratify=readers["user_id"], 
                                   test_size=0.20,
                                   random_state=random_state)
model_evaluator = ModelEvaluator(k_list = [5, 10, 15])


# merged model
results = pd.DataFrame([],
        columns=['modelName',
        'recall@5',
        ' precision@5',
        'f1_score@5',
        'ndcg@5',
        'recall@10',
        'precision@10',
        'f1_score@10',
        'ndcg@10',
        'recall@15',
        'precision@15',
        'f1_score@15',
        'ndcg@15',
        'weight'])





a=1
while a<4:
    b=1
    while b<4:
        c=1
        while c<4:
            print((a,b,c))
            p_model = Popularity_model_merge(art_db,readers,w=(a,b,c))
            cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(p_model, readers, readers_train, readers_test)
            print(f'\nGlobal metrics:\n{cf_global_metrics}')

            v = list(cf_global_metrics.values())
            v.append((1,2,3))
            r1 = pd.DataFrame([v],
                    columns=['modelName',
                    'recall@5',
                    ' precision@5',
                    'f1_score@5',
                    'ndcg@5',
                    'recall@10',
                    'precision@10',
                    'f1_score@10',
                    'ndcg@10',
                    'recall@15',
                    'precision@15',
                    'f1_score@15',
                    'ndcg@15',
                    'weight'])
            c += 1
        b += 1    
    a+= 1
results = results.append(r1,ignore_index=True)
results.to_csv("res.csv", encoding="utf-8", index=False)

