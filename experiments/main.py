#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

#Cleaning
train["Gender"].fillna(train["Gender"].mode()[0], inplace=True)
train["Married"].fillna(train["Married"].mode()[0], inplace=True)
train["Dependents"].fillna(train["Dependents"].mode()[0], inplace=True)
train["Self_Employed"].fillna(train["Self_Employed"].mode()[0], inplace=True)

test["Gender"].fillna(train["Gender"].mode()[0], inplace=True)
test["Married"].fillna(train["Married"].mode()[0], inplace=True)
test["Dependents"].fillna(train["Dependents"].mode()[0], inplace=True)
test["Self_Employed"].fillna(train["Self_Employed"].mode()[0], inplace=True)

train["LoanAmount"].fillna(train["LoanAmount"].median(), inplace=True)
train["Loan_Amount_Term"].fillna(train["Loan_Amount_Term"].mode()[0], inplace=True)
train["Credit_History"].fillna(train["Credit_History"].mode()[0], inplace=True)

test["LoanAmount"].fillna(train["LoanAmount"].median(), inplace=True)
test["Loan_Amount_Term"].fillna(train["Loan_Amount_Term"].mode()[0], inplace=True)
test["Credit_History"].fillna(train["Credit_History"].mode()[0], inplace=True)


#Mapping
train["Gender"] = train["Gender"].map({"Male": 1, "Female": 0})
train["Married"] = train["Married"].map({"Yes": 1, "No": 0})
train["Education"] = train["Education"].map({"Graduate": 1, "Not Graduate": 0})
train["Self_Employed"] = train["Self_Employed"].map({"Yes": 1, "No": 0})
train["Loan_Status"] = train["Loan_Status"].map({"Y": 1, "N": 0})

test["Gender"] = test["Gender"].map({"Male": 1, "Female": 0})
test["Married"] = test["Married"].map({"Yes": 1, "No": 0})
test["Education"] = test["Education"].map({"Graduate": 1, "Not Graduate": 0})
test["Self_Employed"] = test["Self_Employed"].map({"Yes": 1, "No": 0})

train["Dependents"] = train["Dependents"].replace("3+", 3)
test["Dependents"] = test["Dependents"].replace("3+", 3)

train["Dependents"] = train["Dependents"].astype(int)
test["Dependents"] = test["Dependents"].astype(int)

train = pd.get_dummies(train, columns=["Property_Area"], drop_first=True)
test = pd.get_dummies(test, columns=["Property_Area"], drop_first=True)

train.drop("Loan_ID", axis=1, inplace=True)
test.drop("Loan_ID", axis=1, inplace=True)

#Training

x = train.drop("Loan_Status", axis=1)
y = train["Loan_Status"]

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

model = LinearRegression()
model.fit(x_train, y_train)

pred = model.predict(x_val)
msr = mean_squared_error(y_val, pred)
r2score = r2_score(y_val, pred)


print(f"""
Model: {model}
MSR: {msr}
r2_score: {r2score}
""")
