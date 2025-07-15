import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("best_model.pkl", "rb"))

st.set_page_config(page_title="California Housing Price Prediction", layout="centered")
st.title("🏠 California Housing Price Predictor")

st.markdown("Enter the features below to predict the median house value:")

# Input fields
MedInc = st.number_input("Median Income", min_value=0.0, format="%.2f")
HouseAge = st.number_input("House Age", min_value=0.0, format="%.2f")
TotalRooms = st.number_input("Total Rooms", min_value=0.0, format="%.2f")
TotalBedrooms = st.number_input("Total Bedrooms", min_value=0.0, format="%.2f")
Population = st.number_input("Population", min_value=1.0, format="%.2f")
Households = st.number_input("Households", min_value=1.0, format="%.2f")
Latitude = st.number_input("Latitude", format="%.5f")
Longitude = st.number_input("Longitude", format="%.5f")

# Feature engineering
rooms_per_household = TotalRooms / Households

# Input array
input_data = np.array([[MedInc, HouseAge, TotalRooms, TotalBedrooms,
                        Population, Households, Latitude, Longitude,
                        rooms_per_household]])

# Scale input (same structure as training)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_input = scaler.fit_transform(input_data)  # This is placeholder; ideally use same scaler as training

if st.button("Predict"):
    prediction = model.predict(scaled_input)
    st.success(f"🏡 Predicted Median House Value: {prediction[0]:.2f}")
