from some_functions import get_db
from popularity_model import Popularity_model_wo_author, Popularity_model_wt_author
import pandas as pd
import numpy as np 

if __name__ == "__main__":
    # get db with articles
    art_db = get_db(r'C:\Users\a814811\OneDrive - Atos\RecommenderSystem\art_clean_wt_popul_authoroccurences.csv')
    user_db = get_db(r'C:\Users\a814811\OneDrive - Atos\RecommenderSystem\readers.csv')

    # print(art_db['pub_date'])
    User1wt = Popularity_model_wt_author(user_id=5,articles_db=art_db,user_db=user_db)
    User1wo = Popularity_model_wo_author(user_id=5,articles_db=art_db,user_db=user_db)
    User2 = Popularity_model_wo_author(user_id=-3,articles_db=art_db,user_db=user_db)
    User3 = Popularity_model_wo_author(user_id=2,articles_db=art_db,user_db=user_db)
    print(f'{User1wt}\n user ID: {User1wt.user}\n {8} recommendations:{User1wt.recomm(limit=8)}\n')
    print(f'{User1wo}\n user ID: {User1wo.user}\n {8} recommendations:{User1wo.recomm(limit=8)}\n')
    print(f'{User2}\n user ID: {User2.user}\n {8} recommendations:{User2.recomm(limit=8)}\n')
    print(f'{User3}\n user ID: {User3.user}\n {5} recommendations:{User3.recomm(limit=5)}\n')
    