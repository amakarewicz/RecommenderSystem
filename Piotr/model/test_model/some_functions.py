import pandas as pd
import numpy as np
from math import sqrt

def get_db(filename):
    return pd.read_csv(filename)

def prob_vector_from_ratio(ratio):
    ''' zamiana ratio na odcinek prawdopodobienstwa
    :par ratio: krotka stosunku
    :return: lista prawdopodobienstw skumulowanych
    '''
    vect = np.cumsum(ratio)/sum(ratio)
    return vect

def evaluation(ratio):
    # sqrt(sum(r^2))
    se = sqrt(sum( [(it - 1)**2 for it in ratio]))
    return round(se,2)

def choose_recomm(models_recommendations,ratio,limit):
    ''' dla wybranych rekomendacji z modeli wybieramy wg prawdopodobieństwa wyniki z każdego setu
    :par models_recommendations: lista z listami rekomendacji [[r11,r12,r13],[r21,r22,r23],...]
    :par ratio: stosunek wagi poszczegolnych modeli
    '''
    if len(models_recommendations) != len(ratio):
        raise ValueError
    vect = prob_vector_from_ratio(ratio)
    recommendations=[]
    if sum([len(it) for it in models_recommendations]) <= limit:
        # przypadek w którym nie ma wystarczającej liczby artykułów
        for it in models_recommendations:
            recommendations.extend([i for i in it])
        return recommendations 
    else:
        while len(recommendations) < limit:
            p = np.random.rand()
            x = p > vect
            ind = list(x).index(False)
            if len(models_recommendations[ind]) > 0:
                recommendations.append(models_recommendations[ind].pop(0))
            if len(recommendations) != len(set(recommendations)):
                # wyrzucenie powtórek
                recommendations = [it for it in set(recommendations)]
            # print(models_recommendations)
        return recommendations
        

def precision(rec,user_data):
    if len(user_data) != 0 and len(rec) != 0:
        return len([i for i in rec if i in user_data]) / len(rec)
    return np.nan


def recall(rec,user_data):
    if len(user_data) != 0 and len(rec) != 0:
        return len([i for i in rec if i in user_data]) / len(user_data)
    return np.nan

if __name__ == "__main__":
    r = choose_recomm([[], ['ld.1290371', 'ld.144833', 'ld.1288152', 'ld.140939', 'ld.138429', 'ld.1289100']],(2,2),8)
    print(r)
