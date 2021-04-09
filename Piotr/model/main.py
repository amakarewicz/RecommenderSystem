from some_functions import get_db
from popularity_model import Popularity
import pandas as pd
import numpy as np 

if __name__ == "__main__":
    # get db with articles
    art = get_db(r'C:\Users\a814811\OneDrive - Atos\RecommenderSystem\art_clean_wt_popul_authoroccurences.csv')
    User1 = Popularity(1,art)
    #print(User1.art_head())
    print(User1.reccom())