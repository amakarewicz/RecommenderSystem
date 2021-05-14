from some_functions import *
from merged_model import *
from merged_model_sum import *
import pandas as pd
import numpy as np 

# # test
# from period_test import period_eval

art_db = get_db(r'C:\Users\a814811\OneDrive - Atos\RecommenderSystem\art_clean_wt_all_popularity.csv')
art_db = art_db[['nzz_id', 'author', 'department', 'pub_date', 'popularity','department_popularity']]
user_db = get_db(r'C:\Users\a814811\OneDrive - Atos\RecommenderSystem\readers.csv')

m = MergedModelSum(art_db, user_db, w=(1,1))
rec = m.recommend(user_id = 3, limit=15)
print(rec)