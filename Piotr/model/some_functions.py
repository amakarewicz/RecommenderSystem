import pandas as pd
import numpy as np


def get_db(filename):
    return pd.read_csv(filename)

def prob_vector_from_ratio(ratio):
    ''' zamiana ratio na odcinek prawdopodobienstwa
    :par ratio: krotka stosunku
    :return: lista prawdopodobienstw skumulowanych
    '''
    vect = np.cumsum(ratio)/sum(ratio)
    return vect


def choose_recomm(models_recommendations,ratio,limit):
    ''' dla wybranych rekomendacji z modeli wybieramy wg prawdopodobieństwa wyniki z każdego setu
    :par models_recommendations: lista z listami rekomendacji [[r11,r12,r13],[r21,r22,r23],...]
    :par ratio: stosunek wagi poszczegolnych modeli
    '''
    if len(models_recommendations) != len(ratio):
        raise ValueError
    vect = prob_vector_from_ratio(ratio)
    recommendations=[]
    while len(recommendations) < limit:
        p = np.random.rand()
        x = p > vect
        ind = list(x).index(False)
        if len(models_recommendations[ind]) > 0:
            recommendations.append(models_recommendations[ind].pop(0))
        if len(recommendations) != len(set(recommendations)):
            # wyrzucenie powtórek
            recommendations = [it for it in set(recommendations)]
        print(models_recommendations)
    return recommendations
        

if __name__ == "__main__":
    r = choose_recomm([[1,2,3],[4,5,6],[4,8,9]],(1,2,3),5)
    print(r)
