"""mlflow setup utilities"""
import mlflow


def setup_mlflow_experiment():
    """mlflow setup"""
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment(experiment_id=613298809565478243)
