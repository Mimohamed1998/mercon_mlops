"""strealit app"""
import mlflow
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_california_housing

# Load your MLflow model
MODEL_NAME = "xgb_california"
STAGE = "Staging"

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

user_input = {}
for feature in feature_names:
    user_input[feature] = st.sidebar.number_input(f"Enter {feature}:", value=0.0)


# Predict the housing price
if st.sidebar.button("Predict"):
    user_input_df = pd.DataFrame([user_input])

    mlflow.set_tracking_uri(tracking_url)
    loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{STAGE}")
    prediction = loaded_model.predict(user_input_df)

    st.sidebar.subheader("Prediction Result")

    st.sidebar.write(f"Predicted Housing Price: ${prediction[0]*100000:,.2f}")

# Display a sample of the California housing dataset
st.write("Sample of California Housing Dataset:")
st.write(data.head())
