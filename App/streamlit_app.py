import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("models/churn_model.pkl", "rb"))
features = pickle.load(open("models/features.pkl", "rb"))

st.title("Customer Churn Prediction App")

tenure = st.number_input("Tenure")
monthly = st.number_input("Monthly Charges")
total = st.number_input("Total Charges")

input_df = pd.DataFrame([[tenure, monthly, total]],
                        columns=["tenure", "MonthlyCharges", "TotalCharges"])

input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=features, fill_value=0)

if st.button("Predict"):
    pred = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][1]

    if pred == 1:
        st.error(f"⚠️ Churn Risk ({prob:.2f})")
    else:
        st.success(f"✅ No Churn ({prob:.2f})")