from multiprocessing import Pool
from datetime import date, datetime
import pandas as pd
from cf_model_main import CF_model

readers = pd.read_csv("Kamil/data/readers.csv")
readers = readers.rename(columns={"id":"user_id", "art_id":"nzz_id"})


def train(x):
    model = CF_model(user_db = readers)
    return model

if __name__ == '__main__':
    start = datetime.now()
    with Pool(15) as p:
        print(p.map(train, range(10)))
    #for x in range(10):
    #    train(x)
    end = datetime.now()

    print(f"processing took: {end-start}")