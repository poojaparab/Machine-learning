import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import joblib 
from main import regressionmodel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.pipeline import Pipeline
from math import sqrt
import pandas as pd
from load import load_data
import logging
import os
from log import custom_log

logger = custom_log(path="logs/", file="model_training.logs")

app = FastAPI()
@app.get('/')
def health_check():
    return {'status': 'Health check is success!!'}

class PostData(BaseModel):
    test_size: float
    random_state: int
    model_version: int

@app.post("/predict")
def regression_model(post_data:PostData):
    X,y = load_data()
    logger.info("splitting the dataset between train and test")
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = float(post_data.test_size),random_state=int(post_data.random_state))
    regressionmodel(X_train,X_test,y_train,y_test, post_data.test_size, post_data.random_state, post_data.model_version)
    linear_regression_pipe = joblib.load('./models/linear-regression-model_{}.pkl'.format(post_data.model_version))

    y_predicted = linear_regression_pipe.predict(X_test)
    logger.info("predicted value of Y after implementing linear regression {}".format(y_predicted))
    rmse = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted))
    logger.info("Root mean square error is {}".format(rmse))
    score = linear_regression_pipe.score(X_test, y_test)
    logger.info("score of the regression model is: {}".format(score))
    return JSONResponse({"predicted_y": list(y_predicted), "model_rmse":rmse, "score": score})

if __name__=='__main__':
    uvicorn.run(app,port=8000,debug=True,host='0.0.0.0')
