#!/usr/bin/env python3
import pandas as pd

def preprocess_data(train):
	#Cleaning

	train = train.drop("Loan_ID", axis=1, errors="ignore")

	train["Gender"] = train["Gender"].fillna(train["Gender"].mode()[0])
	train["Married"] = train["Married"].fillna(train["Married"].mode()[0])
	train["Dependents"] = train["Dependents"].fillna(train["Dependents"].mode()[0])
	train["Self_Employed"] = train["Self_Employed"].fillna(train["Self_Employed"].mode()[0])

	train["LoanAmount"] = train["LoanAmount"].fillna(train["LoanAmount"].median())
	train["Loan_Amount_Term"] = train["Loan_Amount_Term"].fillna(train["Loan_Amount_Term"].mode()[0])
	train["Credit_History"] = train["Credit_History"].fillna(train["Credit_History"].mode()[0])

	#Mapping
	train["Gender"] = train["Gender"].map({"Male": 1, "Female": 0})
	train["Married"] = train["Married"].map({"Yes": 1, "No": 0})
	train["Education"] = train["Education"].map({"Graduate": 1, "Not Graduate": 0})
	train["Self_Employed"] = train["Self_Employed"].map({"Yes": 1, "No": 0})

	if 'Loan_Status' in train.columns:
		train["Loan_Status"] = train["Loan_Status"].map({"Y": 1, "N": 0})

	train["Dependents"] = train["Dependents"].replace("3+", 3)

	train["Dependents"] = train["Dependents"].astype(int)

	train = pd.get_dummies(train, columns=["Property_Area"], drop_first=True)

	for col in ["Property_Area_Semiurban", "Property_Area_Urban"]:
		if col not in train.columns:
			train[col] = 0

	return train
