from some_functions import get_db
from popularity_model import Popularity
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
    User1 = Popularity(user_id=2,articles_db=art_db,user_db=user_db,art_limit=8)
    User2 = Popularity(user_id=-3,articles_db=art_db,user_db=user_db,art_limit=8)
    User3 = Popularity(user_id=2,articles_db=art_db,art_limit=2)
    print(User1.reccom())
    print(User2.reccom())
    print(User3.reccom())