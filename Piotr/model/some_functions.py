import pandas as pd

def get_db(filename):
    return pd.read_csv(filename)
