#!/usr/bin/env python3

import joblib as jb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from preprocess import preprocess_data
from xgboost import XGBClassifier

df = pd.read_csv("../data/train.csv")
df = preprocess_data(df)

x = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

rf_model = RandomForestClassifier(random_state=42)
logistic_model = LogisticRegression(max_iter=10000)
xgb_model = XGBClassifier()

rf_model.fit(x, y)
logistic_model.fit(x, y)
xgb_model.fit(x, y)

jb.dump(rf_model, "../models/rf_model.pkl")
jb.dump(logistic_model, "../models/logistic_model.pkl")
jb.dump(xgb_model, "../models/xgb_model.pkl")

