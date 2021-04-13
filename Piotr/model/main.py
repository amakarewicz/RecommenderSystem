from some_functions import get_db
from popularity_model import *
import pandas as pd
import numpy as np 

if __name__ == "__main__":
    # get db with articles
    art_db = get_db(r'C:\Users\a814811\OneDrive - Atos\RecommenderSystem\art_clean_wt_all_popularity.csv')
    user_db = get_db(r'C:\Users\a814811\OneDrive - Atos\RecommenderSystem\readers.csv')

    # print(art_db['pub_date'])

    # User1 = Popularity_model(user_id=5,articles_db=art_db,user_db=user_db)
    User1a = Popularity_model_author(user_id=5,articles_db=art_db,user_db=user_db)
    User1d = Popularity_model_department(user_id=5,articles_db=art_db,user_db=user_db)
    # User2 = Popularity_model(user_id=-3,articles_db=art_db,user_db=user_db)
    # User3 = Popularity_model(user_id=2,articles_db=art_db,user_db=user_db)

    # users = [User1d]
    # for User in users:
    #     print(f'{str(User)[18:-30]}\n user ID: {User.user}\n {len(User.recomm(limit=8))} recommendations:\n{User.recomm(limit=8)}\n')
    print(User1a.recomm(limit=10))
    print(User1d.recomm(limit=10))