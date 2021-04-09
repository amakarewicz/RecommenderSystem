from some_functions import get_db
from popularity_model import Popularity
import pandas as pd
import numpy as np 

if __name__ == "__main__":
    # get db with articles
    art_db = get_db(r'C:\Users\a814811\OneDrive - Atos\RecommenderSystem\art_clean_wt_popul_authoroccurences.csv')
    user_db = get_db(r'C:\Users\a814811\OneDrive - Atos\RecommenderSystem\readers.csv')

    # print(user_db.iloc[:,0])
    User1 = Popularity(-3,articles_db=art_db,user_db=user_db)
    print(User1.head(User1.user_db))
    print(User1.reccom())