import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("logistic_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

# User Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])

sex = st.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex == "Male" else 0

age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Number of Siblings/Spouse", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=30.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Dummy encoding for Embarked
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame([[
        pclass,
        sex,
        age,
        sibsp,
        parch,
        fare,
        embarked_q,
        embarked_s
    ]],
    columns=[
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked_Q",
        "Embarked_S"
    ])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Passenger is likely to Survive.\n\nProbability: {probability:.2%}")
    else:
        st.error(f"Passenger is unlikely to Survive.\n\nProbability: {probability:.2%}")