import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("weather_classifier.joblib")

# Streamlit UI
st.title("Weather Classifier")

st.write("Enter weather parameters to classify the weather condition.")

# Example input fields (modify based on your model's input features)
temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=20.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=200.0, value=10.0)

# Make a prediction
if st.button("Predict Weather Condition"):
    features = np.array([[temp, humidity, wind_speed]])
    prediction = model.predict(features)
    st.success(f"Predicted Weather Condition: {prediction[0]}")

