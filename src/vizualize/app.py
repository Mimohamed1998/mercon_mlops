"""strealit app"""
import mlflow
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_california_housing
from mlflow.tracking.client import MlflowClient

# Load your MLflow model
MODEL_NAME = "xgb_california"

# pylint: disable=no-member
housing = fetch_california_housing()
data = pd.DataFrame(housing.data)
data.columns = housing.feature_names
data["PRICE"] = housing.target
data["PRICE"] = data["PRICE"] * 100000

# Title and description
st.title("California Housing Price Prediction")
st.write("This app uses an MLflow model to predict housing prices in California.")

# Input form for user
st.sidebar.header("Input Parameters")

# Define input features
feature_names = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

# Create input fields for user
tracking_url = st.sidebar.text_input("Enter Tracking URI")

# Remove the following line for Production!
current_stage = st.sidebar.selectbox("Stage", ("None","Staging", "Production", "Archived"))

user_input = {}
for feature in feature_names:
    user_input[feature] = st.sidebar.number_input(f"Enter {feature}:", value=0.0)


def get_model_version(models, stage):
    for model in models:
        if model.current_stage == stage:
            model_version = str(model.version)
            break
        model_version = 'No Model Available'
    return model_version


# Predict the housing price
if st.sidebar.button("Predict"):
    user_input_df = pd.DataFrame([user_input])

    mlflow.set_tracking_uri(tracking_url)
    client = MlflowClient(tracking_url)

    loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{current_stage}")
    prediction = loaded_model.predict(user_input_df)

    all_models = client.get_latest_versions("xgb_california")
    VERSION = get_model_version(all_models, current_stage)

    st.sidebar.subheader(f"Prediction Result from model version ${VERSION}")

    st.sidebar.write(f"Predicted Housing Price: ${prediction[0]*100000:,.2f}")

# Display a sample of the California housing dataset
st.write("Sample of California Housing Dataset:")
st.write(data.head())
