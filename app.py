import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler with error handling
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Required file missing: {e}")
    st.stop()

st.set_page_config(page_title="House Price Prediction", layout="centered")
st.title("🏠 House Price Prediction")
st.write("Predict house prices based on Bedrooms, Bathrooms, and Area.")

# ✅ Final consistent order used for everything
FEATURE_COLUMNS = ["Bedrooms", "Bathrooms", "Area"]

# User Inputs → Same order as FEATURE_COLUMNS
bedrooms = st.number_input("Number of Bedrooms", min_value=1, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, value=2)
area = st.number_input("Area (in sqft)", min_value=300, value=1200)

if st.button("Predict"):
    try:
        # ✅ Build DataFrame in correct order
        input_data = pd.DataFrame([[bedrooms, bathrooms, area]], columns=FEATURE_COLUMNS)
        input_scaled = scaler.transform(input_data)
        predicted_price = model.predict(input_scaled)[0]
        st.success(f"Predicted Price: ₹{int(predicted_price):,}")

        # 📈 Generate chart for price vs area (keeping bedrooms & bathrooms fixed)
        st.subheader("📈 Price vs Area (keeping Bedrooms & Bathrooms fixed)")
        area_range = np.arange(300, 3001, 100)
        chart_data = pd.DataFrame(columns=["Area", "Predicted Price"])

        for a in area_range:
            row = pd.DataFrame([[bedrooms, bathrooms, a]], columns=FEATURE_COLUMNS)
            row_scaled = scaler.transform(row)
            price = model.predict(row_scaled)[0]
            chart_data.loc[len(chart_data)] = [a, price]

        st.line_chart(chart_data.set_index("Area"))

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
