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
        return recommendations

def precision(rec,user_data):
    # recommended and user articles
    if len(user_data) != 0 and len(rec) != 0:
        return len([i for i in rec if i in user_data]) / len(rec)
    return np.nan


def recall(rec,user_data):
    # recommended and user articles
    if len(user_data) != 0 and len(rec) != 0:
        return len([i for i in rec if i in user_data]) / len(user_data)
    return np.nan

# tutaj z dodaną wagą
        # def choose_recomm(models_recommendations,ratio,limit,w=(1,1,1)):
        #     ''' dla wybranych rekomendacji z modeli wybieramy wg prawdopodobieństwa wyniki z każdego setu
        #     :par models_recommendations: lista z listami rekomendacji [[r11,r12,r13],[r21,r22,r23],...]
        #     :par ratio: stosunek wagi poszczegolnych modeli
        #     '''
        #     # nadanie wagi
        #     if len(ratio) == 3:
        #         ratio = np.array(ratio)*np.array(w)
        #     if len(models_recommendations) != len(ratio):
        #         raise ValueError
        #     vect = prob_vector_from_ratio(ratio)
        #     recommendations=[]
        #     if sum([len(it) for it in models_recommendations]) <= limit:
        #         # przypadek w którym nie ma wystarczającej liczby artykułów
        #         for it in models_recommendations:
        #             recommendations.extend([i for i in it])
        #         return recommendations 
        #     else:
        #         while len(recommendations) < limit:
        #             p = np.random.rand()
        #             x = p > vect
        #             ind = list(x).index(False)
        #             if len(models_recommendations[ind]) > 0:
        #                 recommendations.append(models_recommendations[ind].pop(0))
        #             if len(recommendations) != len(set(recommendations)):
        #                 # wyrzucenie powtórek
        #                 recommendations = [it for it in set(recommendations)]
        #             # print(models_recommendations)
        #         return recommendations



if __name__ == "__main__":
    r = choose_recomm([
        ['1.18108994', 'ld.1086062', 'ld.153813', 'ld.140509', 'ld.137077', 'ld.150497', 'ld.137081', 'ld.150557'],
        ['ld.155260', 'ld.139338', 'ld.1290435', 'ld.148526', 'ld.142559', 'ld.1290811', 'ld.139047', 'ld.1294764'],
        ['ld.144297', 'ld.139916', 'ld.138179', 'ld.153622', 'ld.154109', 'ld.148355', 'ld.153589', 'ld.138751']],(1,3.61,8),8)
    print(r)
