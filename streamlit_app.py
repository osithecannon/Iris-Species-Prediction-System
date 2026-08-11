
import streamlit as st
import pandas as pd
import joblib

# Load the trained model, scaler, and target names
# Make sure these files are in the same directory as your Streamlit app or provide the full path
try:
    model = joblib.load('iris_best_model.joblib')
    scaler = joblib.load('iris_scaler.joblib')
    target_names = joblib.load('iris_target_names.joblib')
except FileNotFoundError:
    st.error("Error: Model, scaler, or target names file not found. Please ensure 'iris_best_model.joblib', 'iris_scaler.joblib', and 'iris_target_names.joblib' are in the same directory as this app.")
    st.stop()

st.set_page_config(page_title="Iris Species Predictor", layout="centered")

st.title("🌸 Iris Species Predictor")
st.write("Enter the measurements of an Iris flower to predict its species.")

st.sidebar.header('Flower Measurements (cm)')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
    data = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('User Input Measurements')
st.write(df_input)

# Scale the input features
scaled_input = scaler.transform(df_input)
scaled_input_df = pd.DataFrame(scaled_input, columns=df_input.columns)

st.subheader('Scaled Input Features')
st.write(scaled_input_df)

# Make prediction
prediction_proba = model.predict_proba(scaled_input)
prediction = model.predict(scaled_input)

st.subheader('Prediction')
predicted_species = target_names[prediction[0]]
st.write(f"The predicted Iris species is: **{predicted_species}**")

st.subheader('Prediction Probability')
proba_df = pd.DataFrame(prediction_proba, columns=target_names)
st.write(proba_df)

st.write("--- Developed by your AI Assistant --- ")
