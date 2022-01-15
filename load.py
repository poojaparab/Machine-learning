# this file we can for data preporcessing


import pandas as pd
from log import custom_log


logger = custom_log(path="logs/", file="model_training.logs")

def load_data():
    logger.info("Reading car csv file")
    df =pd.read_csv("Data\cardata_headers.csv", index_col="car name")
    logger.info("Data preprocessing to remove wrong values")
    df = df[df.horsepower != '?']
    factors = ['cylinders','displacement','horsepower','acceleration','weight','origin','model year']
    X = pd.DataFrame(df[factors].copy())
    y = df['mpg'].copy()
    return X,y
