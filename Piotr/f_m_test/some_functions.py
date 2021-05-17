import pandas as pd
import numpy as np
from math import sqrt
from typing import Union


def get_db(filename: str) -> pd.DataFrame:
    """ function getting pandas DF from .csv file

    Args:
        filename (str): file / filepath

    Returns:
        [pd.DataFrame]: DF from file
    """
    return pd.read_csv(filename)

def prob_vector_from_ratio(ratio: tuple) -> np.ndarray:
    """ function transforming ratio to cumulative probability vector

    Args:
        ratio (tuple): ratio vector, pe. (1,2,3)

    Returns:
        np.ndarray: cumulative probability vector, pe. (1/6,3/6,6/6)
    """
    vect = np.cumsum(ratio)/sum(ratio)
    return vect

def evaluation(ratio: tuple) -> float:
    """ function giving sqrt(sum(r^2)) where r is ratio vector

    Args:
        ratio (tuple): ratio vector, pe. (1,2,3)
    Returns:
        [float]: y =  sqrt(sum(r^2)) function  where r is ratio vector
    """
    se = sqrt(sum( [(it - 1)**2 for it in ratio]))
    return round(se,2)


def precision(rec: list, user_data: list) -> float:
    """ function giving precision value for list of recommendations and user articles.

    Args:
        rec (list) : articles recommended for user
        user_data (list): articles read by user

    Returns:
        [float, np.nan]: precision value or np.nan if there is no recommendation 
                         or no user articles.
    """
    if len(user_data) != 0 and len(rec) != 0:
        return len([i for i in rec if i in user_data]) / len(rec)
    return np.nan


def recall(rec: list, user_data: list) -> float:
    """ function giving recall value for list of recommendations and user articles.

    Args:
        rec (list) : articles recommended for user
        user_data (list): articles read by user

    Returns:
        [float, np.nan]: recall value or np.nan if there is no recommendation 
                         or no user articles.
    """
    if len(user_data) != 0 and len(rec) != 0:
        return len([i for i in rec if i in user_data]) / len(user_data)
    return np.nan

def f1score(recall: float, precision: float) -> float:
    """ harmonic mean of precision and recall.

    Args:
        recall (float): recall value
        precision (float): precision value

    Returns:
        [float, np.nan] : f1_score value, np.nan if recall or precision is np.nan
    """
    if recall == 0 or precision == 0:
        return 0
    return 2/(1/recall + 1/precision)

def choose_recomm(models_recommendations: list,ratio: tuple, limit: int, w: tuple = (1,1,1)) -> list:
    """ function chosing for selected recommendations from the models, 
        the results from each set according to probability.

    Args:
        models_recommendations (list): list of lists of recommendations 
                                       for each model - [[r11,r12,r13],[r21,r22,r23],...]
        ratio (tuple): default ratio for models
        limit (int): number of recommendations to return
        w (tuple, optional): weight tuple to manipulate ratio. Defaults to (1,1,1).

    Raises:
        ValueError: ratio length is different than number of models.

    Returns:
        list: chosen recommendations according to probability taken from ratio
    """

    # giving weight
    if len(ratio) == 3:
        ratio = np.array(ratio)*np.array(w)
        
    if len(models_recommendations) != len(ratio):
        raise ValueError
    vect = prob_vector_from_ratio(ratio)
    recommendations=[]
    if sum([len(it) for it in models_recommendations]) <= limit:
        # case: no enough articles
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
                # removing duplicates
                recommendations = [it for it in set(recommendations)]
        return recommendations




if __name__ == "__main__":
    r = choose_recomm([
        ['1.18108994', 'ld.1086062', 'ld.153813', 'ld.140509', 'ld.137077', 'ld.150497', 'ld.137081', 'ld.150557'],
        ['ld.155260', 'ld.139338', 'ld.1290435', 'ld.148526', 'ld.142559', 'ld.1290811', 'ld.139047', 'ld.1294764'],
        ['ld.144297', 'ld.139916', 'ld.138179', 'ld.153622', 'ld.154109', 'ld.148355', 'ld.153589', 'ld.138751']],(1,3.61,8),8)
    print(r)
