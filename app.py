
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Your trained model
import pickle
model = pickle.load(open('model.pkl', 'rb'))

st.title("House Price Prediction")

bedrooms = st.number_input("Number of Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Number of Bathrooms", 1, 10, 2)
area = st.number_input("Area (in sqft)", 300, 10000, 1200)

if st.button("Predict"):
    input_data = pd.DataFrame([[bedrooms, bathrooms, area]], columns=["Bedrooms", "Bathrooms", "Area"])
    price = model.predict(input_data)[0]
    st.success(f"Predicted Price: ₹{int(price):,}")

    # Chart: Price vs Area
    st.subheader("📊 Price vs Area (keeping bedrooms and bathrooms fixed)")

    # Generate area range
    area_range = np.arange(500, 3001, 100)
    price_predictions = []

    for a in area_range:
        row = pd.DataFrame([[bedrooms, bathrooms, a]], columns=["Bedrooms", "Bathrooms", "Area"])
        p = model.predict(row)[0]
        price_predictions.append(p)

    chart_data = pd.DataFrame({
        "Area (sqft)": area_range,
        "Predicted Price": price_predictions
    })

    st.line_chart(chart_data.set_index("Area (sqft)"))

