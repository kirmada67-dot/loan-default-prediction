#!/usr/bin/env python3

import streamlit as st
import pandas as pd
from src.predict import classify_user

st.set_page_config(page_title="Loan Default Prediction")

st.title("Loan Default Prediction")
st.write("Enter applicant details to predict loan approval.")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0,
    value=5000,
    step=100
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0,
    value=0,
    step=100
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=150,
    step=1
)

loan_term = st.number_input(
    "Loan Amount Term (months)",
    min_value=0,
    value=360,
    step=12
)

credit_history = st.selectbox("Credit History", [1, 0])

property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

if st.button("Predict"):

    data = pd.DataFrame([{
        "Loan_ID": "LP000000",
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])

    prediction = classify_user(data)

    if prediction == "Approved":
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")
