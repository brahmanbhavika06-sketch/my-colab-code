import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing files
model = joblib.load("LR_ford_car.pkl")
scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")

# Page settings
st.set_page_config(page_title="Ford Car Price Predictor", layout="centered")

st.title("Ford Car Price Predictor")
st.write("Enter the car details below to predict the selling price.")

# -----------------------------
# User Inputs
# -----------------------------
year = st.number_input("Year", min_value=1990, max_value=2025, value=2018)

mileage = st.number_input("Mileage", min_value=0, value=30000)

tax = st.number_input("Tax", min_value=0, value=145)

mpg = st.number_input("MPG", min_value=0.0, value=55.4)

engineSize = st.number_input("Engine Size", min_value=0.0, value=1.5)

model_name = st.selectbox(
    "Model",
    [
        "Fiesta",
        "Focus",
        "Kuga",
        "EcoSport",
        "Mondeo",
        "Ka+",
        "Puma",
        "B-MAX",
        "C-MAX",
        "S-MAX",
        "Galaxy",
        "Edge",
        "Grand C-MAX",
        "Mustang",
        "Ranger",
        "Transit Tourneo"
    ]
)

transmission = st.selectbox(
    "Transmission",
    [
        "Manual",
        "Automatic",
        "Semi-Auto"
    ]
)

fuelType = st.selectbox(
    "Fuel Type",
    [
        "Petrol",
        "Diesel",
        "Hybrid",
        "Electric",
        "Other"
    ]
)




# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):

    # Create input dictionary
    input_data = {
        "year": year,
        "mileage": mileage,
        "tax": tax,
        "mpg": mpg,
        "engineSize": engineSize
    }

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale numerical columns
    numerical_cols = ["year", "mileage", "tax", "mpg", "engineSize"]
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Create empty dataframe with all encoded columns
    final_df = pd.DataFrame(columns=encoded_columns)
    final_df.loc[0] = 0

    # Fill numerical columns
    for col in numerical_cols:
        if col in final_df.columns:
            final_df.loc[0, col] = input_df.loc[0, col]

    # One-Hot Encoding
    model_col = "model_" + model_name
    transmission_col = "transmission_" + transmission
    fuel_col = "fuelType_" + fuelType

    if model_col in final_df.columns:
        final_df.loc[0, model_col] = 1

    if transmission_col in final_df.columns:
        final_df.loc[0, transmission_col] = 1

    if fuel_col in final_df.columns:
        final_df.loc[0, fuel_col] = 1

    # Prediction
    prediction = model.predict(final_df)[0]

    st.success(f"Predicted Price: £ {prediction:,.2f}")
    