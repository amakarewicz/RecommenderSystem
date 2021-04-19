from popularity_model import *
from some_functions import get_db
from some_functions import choose_recomm
from numpy import nan

def submodel_merge(user_id,art_db,limit=10,user_db=None):
    User = Popularity_model(user_id=user_id,articles_db=art_db,user_db=user_db)
    User_a = Popularity_model_author(user_id=user_id,articles_db=art_db,user_db=user_db)
    User_d = Popularity_model_department(user_id=user_id,articles_db=art_db,user_db=user_db)
    all_recommendations = []
    ratio = []
    # można dodać dodatkowe wagi dla submodeli w tym miejscu
    for u in [User,User_a,User_d]:
        rec, _, ev = u.recomm(limit=limit)
        all_recommendations.append(rec)
        ratio.append(ev)
    user_articles = User.recomm(limit=limit)[1]
    recommended = choose_recomm(all_recommendations,ratio,limit)
    return recommended, user_articles, nan

if __name__ == "__main__":
    art_db = get_db(r'C:\Users\a814811\OneDrive - Atos\RecommenderSystem\art_clean_wt_all_popularity.csv')
    user_db = get_db(r'C:\Users\a814811\OneDrive - Atos\RecommenderSystem\readers.csv')
    x = submodel_merge(user_id=-1,art_db=art_db,limit=10,user_db=user_db)
    print(x)