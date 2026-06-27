#!/usr/bin/env python3

import pandas as pn
import joblib as jb
from preprocess import preprocess_data

model = jb.load("../models/logistic_model.pkl")
FEATURE_ORDER = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area_Semiurban',
       'Property_Area_Urban']

def classify_user(input_data):
	df = preprocess_data(input_data)
	df = df[FEATURE_ORDER]
	prediction = model.predict(df)
	if prediction[0] == 1:
		return "Approved"
	else:
		return "Rejected"

if __name__ == "__main__":

	sample_data = pn.DataFrame([{"Loan_ID":"LP999998","Gender":"Female","Married":"No","Dependents":"0","Education":"Not Graduate","Self_Employed":"Yes","ApplicantIncome":1200,"CoapplicantIncome":0,"LoanAmount":250,"Loan_Amount_Term":360,"Credit_History":0,"Property_Area":"Urban"}])
	prediction = classify_user(sample_data)
	print("Loan Status: ",prediction)
