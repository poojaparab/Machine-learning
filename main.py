# import uvicorn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.pipeline import Pipeline
from math import sqrt
import joblib 
from log import custom_log

logger = custom_log(path="logs/", file="model_training.logs")


def regressionmodel(X_train,X_test,y_train,y_test, test_size, random_state, model_version):
    logger.info("Implementing standard scalar for data scaling and linear regression model")
    
    linear_regression_pipe = Pipeline([('scl', StandardScaler()),('clf',LinearRegression())])
    linear_regression_pipe.fit(X_train, y_train)
    logger.info("Creating pickle file for given model version")
    joblib.dump(linear_regression_pipe, './models/linear-regression-model_{}.pkl'.format(model_version))


