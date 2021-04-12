from some_functions import get_db
from popularity_model import *
import pandas as pd
import numpy as np 

if __name__ == "__main__":
    # get db with articles
    art_db = get_db(r'C:\Users\a814811\OneDrive - Atos\RecommenderSystem\art_clean_wt_popul_authoroccurences.csv')
    user_db = get_db(r'C:\Users\a814811\OneDrive - Atos\RecommenderSystem\readers.csv')

    # print(user_db.iloc[:,0])
    # x = user_db[user_db['art_id'] == '1.18108994']
    # print(x)
    # x = x.iloc[:,1].tolist()
    # print(user_db[user_db['id'] == 3].iloc[:,1].tolist())
    User1 = Popularity_model_wo_author(user_id=2,articles_db=art_db,user_db=user_db)
    User2 = Popularity_model_wo_author(user_id=-3,articles_db=art_db,user_db=user_db)
    User3 = Popularity_model_wo_author(user_id=2,articles_db=art_db)
    print(User1.recomm(limit=8))
    print(User2.recomm(limit=8))
    print(User3.recomm(limit=8))