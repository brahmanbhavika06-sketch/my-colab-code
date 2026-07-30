
import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Multi-Model ML App",
    page_icon="🤖"
)

st.title("🤖 Multi-Model Machine Learning App")
st.write("Classification and Regression Prediction")

# Load models
classification_models = joblib.load("classification_models.pkl")
regression_models = joblib.load("regression_models.pkl")

classification_scaler = joblib.load("classification_scaler.pkl")
regression_scaler = joblib.load("regression_scaler.pkl")

classification_columns = joblib.load("classification_columns.pkl")
regression_columns = joblib.load("regression_columns.pkl")

problem_type = st.selectbox(
    "Select Problem Type",
    ["Classification", "Regression"]
)

if problem_type == "Classification":

    st.header("Classification")

    algorithm = st.selectbox(
        "Select Classification Algorithm",
        list(classification_models.keys())
    )

    inputs = []

    for column in classification_columns:
        value = st.number_input(column, value=0.0)
        inputs.append(value)

    if st.button("Predict Classification"):

        input_data = np.array(inputs).reshape(1, -1)
        input_scaled = classification_scaler.transform(input_data)

        model = classification_models[algorithm]
        prediction = model.predict(input_scaled)[0]

        if prediction == 0:
            result = "Malignant"
        else:
            result = "Benign"

        st.success(f"Prediction: {result}")

else:

    st.header("Regression")

    algorithm = st.selectbox(
        "Select Regression Algorithm",
        list(regression_models.keys())
    )

    inputs = []

    for column in regression_columns:
        value = st.number_input(column, value=0.0)
        inputs.append(value)

    if st.button("Predict House Value"):

        input_data = np.array(inputs).reshape(1, -1)
        input_scaled = regression_scaler.transform(input_data)

        model = regression_models[algorithm]
        prediction = model.predict(input_scaled)[0]

        st.success(
            f"Predicted House Value: {prediction:.4f}"
        )
