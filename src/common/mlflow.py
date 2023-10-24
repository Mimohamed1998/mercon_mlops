"""mlflow setup utilities"""
import mlflow


def setup_mlflow_experiment(url=None, exp_id=None):
    """mlflow setup"""
    mlflow.set_tracking_uri(url)
    mlflow.set_experiment(experiment_id=exp_id)
