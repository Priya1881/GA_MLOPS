# load the train and test
# train algo
# save the metrices, params
import os
import datetime
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from get_data import read_params
import argparse
import pickle
from Application_Logging.logger import App_Logger

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def run_xgb(x1, y_train):
    model_xgb = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators = 1000,learning_rate=0.5,max_depth=8)
    model_xgb.fit(x1, y_train)
    return  model_xgb


def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    model_dir = config["model_dir"]
    file_object = open('Training_log.txt', 'a+')
    logger = App_Logger()

    df=pd.read_csv(train_data_path) #Reading the processed dataset

    df["date"] = pd.to_datetime(df["date"]).dt.date
    X_train = df[df['date'] <= datetime.date(2017, 5, 31)] #splitting the dataset based on date for trainging data
    val_X = df[df['date'] > datetime.date(2017, 5, 31)] #spliting the dataset based on date for validation data
    logger.log(file_object,"Splitting dataset completed")

    X_train = X_train.drop(['date'], axis=1)
    val_X = val_X.drop(['date'], axis=1)

    y_train = np.log1p((X_train["transactionRevenue"]).values)
    val_y = np.log1p((val_X["transactionRevenue"]).values)
    logger.log(file_object, "Log transformation of transaction Revenue values completed")
    x1 = X_train.drop(['transactionRevenue'], axis=1)
    val_x1 = val_X.drop(['transactionRevenue'], axis=1)
    y_train = pd.DataFrame(y_train)
    val_y = pd.DataFrame(val_y)

    xgb_model=run_xgb(x1,y_train)
    prediction = xgb_model.predict(val_x1)
    (rmse,mae,r2)=eval_metrics(val_y,prediction)
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    pickle.dump(xgb_model, open(model_path, 'wb'))

   ##############################
    logger.log(file_object, "Model file created successfully")
    file_object.close()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)