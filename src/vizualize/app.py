import streamlit as st
import pandas as pd
import mlflow

# Load your MLflow model
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

model_name = "xgb_california"
stage = "Staging"

loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")


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
    "Longitude"
]

# Create input fields for user
user_input = {}
for feature in feature_names:
    user_input[feature] = st.sidebar.number_input(f"Enter {feature}:", value=0.0)

# Predict the housing price
if st.sidebar.button("Predict"):
    user_input_df = pd.DataFrame([user_input])
    prediction = loaded_model.predict(user_input_df)
    st.sidebar.subheader("Prediction Result")
    st.sidebar.write(f"Predicted Housing Price: ${prediction[0]:,.2f}")

# Display a sample of the California housing dataset
st.write("Sample of California Housing Dataset:")
california_housing = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv")
st.write(california_housing.head())
