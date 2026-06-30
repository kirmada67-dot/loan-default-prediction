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

	sample_data = pn.DataFrame([{"Loan_ID":"LP999999","Gender":"Male","Married":"Yes","Dependents":"2","Education":"Graduate","Self_Employed":"No","ApplicantIncome":5500,"CoapplicantIncome":1800,"LoanAmount":150,"Loan_Amount_Term":360,"Credit_History":1,"Property_Area":"Semiurban"}])
	prediction = classify_user(sample_data)
	print("Loan Status: ",prediction)


