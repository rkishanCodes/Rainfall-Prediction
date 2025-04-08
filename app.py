import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Rainfall Prediction App", layout="centered")

st.title("🌦️ Rainfall Prediction App")
st.write("Enter the weather data and choose a model to predict whether it will rain today.")

# Load all models
models = {
    "Logistic Regression": pickle.load(open("models/Logistic_Regression.pkl", "rb")),
    "Random Forest": pickle.load(open("models/Random_Forest.pkl", "rb")),
    "Gradient Boosting": pickle.load(open("models/Gradient_Boosting.pkl", "rb")),
    "SVM": pickle.load(open("models/Support_Vector_Machine.pkl", "rb")),
    "XGBoost": pickle.load(open("models/XGBoost.pkl", "rb"))
}

# Define input fields
def user_input():
    data = {
        "MinTemp": st.number_input("Min Temperature (°C)", value=10.0),
        "MaxTemp": st.number_input("Max Temperature (°C)", value=25.0),
        "Rainfall": st.number_input("Rainfall (mm)", value=0.0),
        "Evaporation": st.number_input("Evaporation (mm)", value=5.0),
        "Sunshine": st.number_input("Sunshine (hrs)", value=7.0),
        "WindGustDir": st.selectbox("Wind Gust Direction", options=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']),
        "WindGustSpeed": st.number_input("Wind Gust Speed (km/h)", value=35.0),
        "WindDir9am": st.selectbox("Wind Direction at 9am", options=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']),
        "WindDir3pm": st.selectbox("Wind Direction at 3pm", options=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']),
        "WindSpeed9am": st.number_input("Wind Speed at 9am (km/h)", value=10.0),
        "WindSpeed3pm": st.number_input("Wind Speed at 3pm (km/h)", value=15.0),
        "Humidity9am": st.number_input("Humidity at 9am (%)", value=60.0),
        "Humidity3pm": st.number_input("Humidity at 3pm (%)", value=50.0),
        "Pressure9am": st.number_input("Pressure at 9am (hPa)", value=1010.0),
        "Pressure3pm": st.number_input("Pressure at 3pm (hPa)", value=1005.0),
        "Cloud9am": st.slider("Cloud at 9am (oktas)", min_value=0, max_value=8, value=3),
        "Cloud3pm": st.slider("Cloud at 3pm (oktas)", min_value=0, max_value=8, value=4),
        "Temp9am": st.number_input("Temperature at 9am (°C)", value=15.0),
        "Temp3pm": st.number_input("Temperature at 3pm (°C)", value=20.0),
        "RainToday": st.selectbox("Did it rain today?", options=['No', 'Yes']),
    }

    # Derived features
    data["Humidity_Diff"] = data["Humidity9am"] - data["Humidity3pm"]
    data["Temp_Diff"] = data["Temp9am"] - data["Temp3pm"]
    data["Pressure_Diff"] = data["Pressure9am"] - data["Pressure3pm"]
    data["WindSpeed_Ratio"] = data["WindSpeed9am"] / (data["WindSpeed3pm"] + 1)

    return pd.DataFrame([data])

# Select model
selected_model = st.selectbox("Choose the prediction model", list(models.keys()))

input_df = user_input()

if st.button("Predict Rainfall"):
    model = models[selected_model]
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

    st.subheader(f"Prediction using {selected_model}:")
    st.write("🌧️ **Rain Expected**" if prediction == 1 else "☀️ **No Rain Expected**")

    if prediction_proba is not None:
        st.subheader("Probability of Rain:")
        st.progress(prediction_proba)
        st.write(f"{prediction_proba*100:.2f}%")