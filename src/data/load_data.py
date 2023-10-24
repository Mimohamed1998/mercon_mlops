"""load data for the exp"""
import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_cali_house_data():
    """
    Loads california housing dataset from skleran and convert it to pandas data frame
    """
    # pylint: disable=no-member
    housing = fetch_california_housing()
    data = pd.DataFrame(housing.data)
    data.columns = housing.feature_names
    data["PRICE"] = housing.target
    return data


def get_features_and_labels(data):
    """return as two x and y datasets"""
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    return x, y
