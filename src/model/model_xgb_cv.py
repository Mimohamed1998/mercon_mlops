"""run cross validation xgb"""
# import sys
# sys.path.append(r"/Users/vinuraperera/Documents/mercon_mlops/src/")
import mlflow
import xgboost as xgb
from common.mlflow import setup_mlflow_experiment
from data.load_data import load_cali_house_data, get_features_and_labels

setup_mlflow_experiment('https://35a0-35-222-156-22.ngrok-free.app/','790539312356347373')
mlflow.autolog(exclusive=False)

with mlflow.start_run():
    data = load_cali_house_data()
    X, y = get_features_and_labels(data) # pylint: disable=duplicate-code
    data_dmatrix = xgb.DMatrix(data=X, label=y)

    params = {
        "objective": "reg:linear",
        "colsample_bytree": 0.3,
        "learning_rate": 0.1,
        "max_depth": 6,
        "alpha": 10,
    }

    cv_results = xgb.cv(
        dtrain=data_dmatrix,
        params=params,
        nfold=3,
        num_boost_round=50,
        early_stopping_rounds=10,
        metrics="rmse",
        as_pandas=True,
        seed=123,
    )

    test_rmse = cv_results["test-rmse-mean"].tail(1)
    xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

    mlflow.log_metric("root_mean_squared_error", test_rmse)
