import pandas as pd
import csv

dataframe = pd.read_json(r'C:\Users\a814811\OneDrive - Atos\Documents\dane\articles.json',lines=True)

print(dataframe.columns())

