import sys
sys.path.append(r"C:/workspace/mercon_mlops/src")

import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score
from data.load_data import load_cali_house_data
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
from common.mlflow import setup_mlflow_experiment

setup_mlflow_experiment()
mlflow.autolog(exclusive=False)

with mlflow.start_run():
    X, y = load_cali_house_data()
    data_dmatrix = xgb.DMatrix(data=X,label=y)

    params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 6, 'alpha': 10}

    cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
    

    test_rmse = cv_results["test-rmse-mean"].tail(1)
    xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

    mlflow.log_metric("root_mean_squared_error", test_rmse)
