import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("weather_classifier.joblib")

# Streamlit UI
st.title("Weather Classifier")

st.write("Enter weather parameters to classify the weather condition.")

# Input fields for each feature
temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=20.0)
humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
dewpoint = st.number_input("Dewpoint (°C)", min_value=-50.0, max_value=50.0, value=10.0)
precipitation = st.number_input("Precipitation (mm)", min_value=0.0, max_value=500.0, value=0.0)
cloudcover = st.number_input("Cloud Cover (%)", min_value=0.0, max_value=100.0, value=50.0)
et0 = st.number_input("Reference Evapotranspiration (mm)", min_value=0.0, max_value=500.0, value=0.0)
windspeed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=200.0, value=10.0)
winddirection = st.number_input("Wind Direction (°)", min_value=0.0, max_value=360.0, value=180.0)
soil_temp = st.number_input("Soil Temperature (°C)", min_value=-50.0, max_value=50.0, value=15.0)
weather = st.selectbox("Observed Weather Condition", ["Sun", "Rain", "Cloudy", "Windy", "Fog", "Snow"])

# Make a prediction
if st.button("Predict Weather Condition"):
    # Prepare the features
    features = np.array([[temp, humidity, dewpoint, precipitation, cloudcover, et0, windspeed, winddirection, soil_temp]])

    # Predict the weather condition
    prediction = model.predict(features)
    
    # Display the result
    st.success(f"Predicted Weather Condition: {prediction[0]}")
