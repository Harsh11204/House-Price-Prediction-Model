import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model and scaler safely
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction")
st.write("Predict house prices based on Bedrooms, Bathrooms, and Area.")

# Consistent feature order for both scaling and prediction
FEATURE_COLUMNS = ["Area", "Bathrooms", "Bedrooms"]

# User Inputs
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
area = st.number_input("Area (in sqft)", min_value=300, max_value=10000, value=1200)

if st.button("Predict"):
    try:
        input_data = pd.DataFrame([[area, bathrooms, bedrooms]], columns=FEATURE_COLUMNS)
        input_scaled = scaler.transform(input_data)
        predicted_price = model.predict(input_scaled)[0]
        st.success(f"Predicted Price: ₹{int(predicted_price):,}")

        # 📈 Price vs Area Chart
        st.subheader("📈 Price vs Area (keeping Bedrooms & Bathrooms fixed)")

        # Use meaningful range based on your **training data's min/max** (adjust if needed)
        area_range = np.arange(300, 3001, 100)
        prices = []

        for a in area_range:
            row = pd.DataFrame([[a, bathrooms, bedrooms]], columns=FEATURE_COLUMNS)
            row_scaled = scaler.transform(row)
            price = model.predict(row_scaled)[0]
            prices.append(price)

        chart_data = pd.DataFrame({
            "Area (sqft)": area_range,
            "Predicted Price": prices
        })

        st.line_chart(chart_data.set_index("Area (sqft)"))

    except Exception as e:
        st.error(f"An error occurred: {e}")
