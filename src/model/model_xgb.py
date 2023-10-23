"""run xgb"""
import mlflow
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from common.mlflow import setup_mlflow_experiment
from data.load_data import load_cali_house_data

setup_mlflow_experiment()
mlflow.autolog(exclusive=False)

with mlflow.start_run():
    X, y = load_cali_house_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    xg_reg = xgb.XGBRegressor(
        objective="reg:linear",
        colsample_bytree=0.3,
        learning_rate=0.1,
        max_depth=5,
        alpha=10,
        n_estimators=10,
    )

    xg_reg.fit(X_train, y_train)

    preds = xg_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    mlflow.log_metric("root_mean_squared_error", rmse)
