#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sbn
import pandas as pd

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

sbn.countplot(x='Loan_Status', data=train)
plt.title("Loan Status Distribution")
plt.show()

sbn.countplot(
    x='Credit_History',
    hue='Loan_Status',
    data=train
)

plt.title("Credit History vs Loan Status")

plt.show()

sbn.histplot(
    train['ApplicantIncome'],
    kde=True
)

plt.title("Applicant Income Distribution")

plt.show()

plt.figure(figsize=(10,6))

sbn.heatmap(
    train.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()
