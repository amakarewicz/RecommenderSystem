import pandas as pd
import numpy as np 


class Popularity:
    '''
    Popularity object contains list of <art limit > 
    recommended articles based on user and articles database
    :param user_id: user id
    :type arg: str
    :param articles_db: user database
    :type arg: pandas table
    :param art_limit: number of reccomended articles
    :type arg: int
    '''
    def __init__(self,user_id,articles_db,art_limit):
        self.users = user_id
        self.articles = articles_db
        self.art_limit=art_limit
        self.reccomended_articles = None
    