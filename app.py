
import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("House Price Prediction")
st.write("Predict house prices based on Bedrooms, Bathrooms, and Area.")

bedrooms = st.number_input("Number of Bedrooms", min_value=0, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=0, value=2)
area = st.number_input("Area (in sqft)", min_value=0, value=1000)

if st.button("Predict"):
    input_data = np.array([[bedrooms, bathrooms, area]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Price: ₹{int(prediction[0])}")
