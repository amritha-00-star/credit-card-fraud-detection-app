import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import glob
import os
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Fraud Detection App", layout="centered")

st.title("Fraud Detection App")

# 1. Dynamically Load Dataset & Train Model
@st.cache_resource
def load_and_train():
    path = kagglehub.dataset_download("ealaxi/paysim1")
    
    # Automatically locate the CSV regardless of its exact file name
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        st.error("No CSV file found in dataset folder.")
        st.stop()
        
    df = pd.read_csv(csv_files[0])
    
    # Map transaction types
    type_map = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3, "CASH_IN": 4}
    df['type'] = df['type'].map(type_map)
    
    # Target and features
    feature_cols = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    
    # Create balanced dataset
    fraud = df[df.isFraud == 1]
    legit = df[df.isFraud == 0].sample(n=len(fraud), random_state=42)
    balanced_df = pd.concat([legit, fraud], axis=0)
    
    X_balanced = balanced_df[feature_cols]
    y_balanced = balanced_df['isFraud']
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_balanced, y_balanced)
    
    return model

with st.spinner("Downloading dataset and training Logistic Regression model..."):
    model = load_and_train()

# 2. Input Fields
type_options = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
type_map = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3, "CASH_IN": 4}

transaction_type = st.selectbox("Transaction Type", type_options)
amount = st.number_input("Amount", value=1000.00, step=100.00)
oldbalanceOrg = st.number_input("Old Balance (Sender)", value=10000.00, step=100.00)
newbalanceOrig = st.number_input("New Balance (Sender)", value=9000.00, step=100.00)
oldbalanceDest = st.number_input("Old Balance (Receiver)", value=0.00, step=100.00)
newbalanceDest = st.number_input("New Balance (Receiver)", value=0.00, step=100.00)

if st.button("Predict"):
    type_num = type_map[transaction_type]
    input_data = np.array([[type_num, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]])
    
    prediction = model.predict(input_data)[0]
    
    st.markdown(f"### Prediction : '{prediction}'")
    
    if prediction == 1:
        st.error("🚨 This transaction looks like a FRAUD!")
    else:
        st.success("✅ This transaction looks like it is NOT a fraud.")
