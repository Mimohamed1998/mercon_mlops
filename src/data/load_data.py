from sklearn.datasets import fetch_california_housing
import pandas as pd


def load_cali_house_data():
    """
    Loads california housing dataset from skleran and convert it to pandas data frame
    """
    housing = fetch_california_housing()
    data = pd.DataFrame(housing.data)
    data.columns = housing.feature_names
    data['PRICE'] = housing.target
    x, y = data.iloc[:,:-1],data.iloc[:,-1]

    return x, y