#!/usr/bin/env python3
import pandas as pd


train = pd.read_csv("../data/train.csv")
print(train.isna().sum())
print(train["Loan_Status"].value_counts())
