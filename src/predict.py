#!/usr/bin/env python3

import pandas as pn
from inference import classify_user

sample_data = pn.DataFrame([{"Loan_ID":"LP999998","Gender":"Female","Married":"No","Dependents":"0","Education":"Not Graduate","Self_Employed":"Yes","ApplicantIncome":1200,"CoapplicantIncome":0,"LoanAmount":250,"Loan_Amount_Term":360,"Credit_History":0,"Property_Area":"Urban"}])

prediction = classify_user(sample_data)

print("Loan Status:", prediction)
