#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sbn
import pandas as pd

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

sbn.countplot(x='Loan_Status', data=train)
plt.title("Loan Status Distribution")
plt.show()

