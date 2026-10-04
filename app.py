import glob
import os
import kagglehub
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression

# Page Configuration
st.set_page_config(
    page_title="Enterprise Fraud Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
)


# Load Dataset & Train Model
@st.cache_resource
def load_data_and_train():
  path = kagglehub.dataset_download("ealaxi/paysim1")
  csv_files = glob.glob(os.path.join(path, "*.csv"))
  if not csv_files:
    st.error("No CSV file found.")
    st.stop()

  df = pd.read_csv(csv_files[0])

  type_map = {
      "PAYMENT": 0,
      "TRANSFER": 1,
      "CASH_OUT": 2,
      "DEBIT": 3,
      "CASH_IN": 4,
  }
  df["type_num"] = df["type"].map(type_map)

  feature_cols = [
      "type_num",
      "amount",
      "oldbalanceOrg",
      "newbalanceOrig",
      "oldbalanceDest",
      "newbalanceDest",
  ]

  # Balanced Dataset Training
  fraud = df[df.isFraud == 1]
  legit = df[df.isFraud == 0].sample(n=len(fraud), random_state=42)
  balanced_df = pd.concat([legit, fraud], axis=0)

  X_balanced = balanced_df[feature_cols]
  y_balanced = balanced_df["isFraud"]

  model = LogisticRegression(max_iter=1000)
  model.fit(X_balanced, y_balanced)

  return df, model


with st.spinner("Initializing Fraud Detection Engine & Analytics..."):
  df, model = load_data_and_train()

# Sidebar Navigation Menu
st.sidebar.title("🛡️ Fraud Engine Menu")
page = st.sidebar.radio(
    "Navigate to:",
    ["📊 Executive Analytics", "🚨 Real-Time Predictor", "📁 Batch CSV Scanner"],
)

# PAGE 1: EXECUTIVE ANALYTICS
if page == "📊 Executive Analytics":
  st.title("📊 Financial Fraud Analytics Dashboard")
  st.caption("Comprehensive risk insights across historical transaction data.")

  col1, col2, col3, col4 = st.columns(4)
  total_tx = len(df)
  total_fraud = int(df["isFraud"].sum())
  fraud_rate = (total_fraud / total_tx) * 100
  fraud_amount = df[df["isFraud"] == 1]["amount"].sum()

  col1.metric("Total Transactions", f"{total_tx:,}")
  col2.metric("Detected Fraud Cases", f"{total_fraud:,}")
  col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
  col4.metric("Total Monetary Loss", f"${fraud_amount:,.2f}")

  st.divider()

  chart_col1, chart_col2 = st.columns(2)

  with chart_col1:
    st.subheader("Fraud Distribution by Transaction Type")
    fraud_type = df[df["isFraud"] == 1]["type"].value_counts().reset_index()
    fraud_type.columns = ["Transaction Type", "Fraud Count"]
    fig1 = px.bar(
        fraud_type,
        x="Transaction Type",
        y="Fraud Count",
        color="Transaction Type",
        text_auto=True,
    )
    st.plotly_chart(fig1, use_container_width=True)

  with chart_col2:
    st.subheader("Legitimate vs Fraud Volume Comparison")
    status_df = pd.DataFrame({
        "Status": ["Legitimate", "Fraud"],
        "Count": [total_tx - total_fraud, total_fraud],
    })
    fig2 = px.pie(
        status_df,
        values="Count",
        names="Status",
        hole=0.4,
        color_discrete_sequence=["#2ecc71", "#e74c3c"],
    )
    st.plotly_chart(fig2, use_container_width=True)

  st.subheader("Transaction Amount Distribution (Fraud vs Legitimate)")
  fig3 = px.box(
      df.sample(20000, random_state=42),
      x="isFraud",
      y="amount",
      color="isFraud",
      labels={"isFraud": "Is Fraud (1=Yes, 0=No)", "amount": "Amount ($)"},
      log_y=True,
  )
  st.plotly_chart(fig3, use_container_width=True)

# PAGE 2: REAL-TIME PREDICTOR
elif page == "🚨 Real-Time Predictor":
  st.title("🚨 Single Transaction Classification")
  st.caption("Submit specific parameters to detect real-time fraud risks.")

  type_options = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
  type_map = {
      "PAYMENT": 0,
      "TRANSFER": 1,
      "CASH_OUT": 2,
      "DEBIT": 3,
      "CASH_IN": 4,
  }

  c1, c2 = st.columns(2)

  with c1:
    transaction_type = st.selectbox("Transaction Type", type_options)
    amount = st.number_input("Transaction Amount ($)", value=1000.00, step=100.0)
    oldbalanceOrg = st.number_input(
        "Sender Initial Balance ($)", value=10000.00, step=100.0
    )

  with c2:
    newbalanceOrig = st.number_input(
        "Sender New Balance ($)", value=9000.00, step=100.0
    )
    oldbalanceDest = st.number_input(
        "Receiver Initial Balance ($)", value=0.00, step=100.0
    )
    newbalanceDest = st.number_input(
        "Receiver New Balance ($)", value=0.00, step=100.0
    )

  if st.button("Evaluate Transaction Risk", type="primary"):
    type_num = type_map[transaction_type]
    input_data = np.array([[
        type_num,
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest,
    ]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.divider()
    res_c1, res_c2 = st.columns(2)

    with res_c1:
      if prediction == 1:
        st.error("🚨 **Classification:** FRAUD DETECTED")
      else:
        st.success("✅ **Classification:** LEGITIMATE TRANSACTION")

    with res_c2:
      st.metric("Fraud Probability", f"{prob * 100:.2f}%")

# PAGE 3: BATCH CSV SCANNER
elif page == "📁 Batch CSV Scanner":
  st.title("📁 Bulk Fraud File Analysis")
  st.caption("Upload a CSV file of transactions for automated classification.")

  uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

  if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.write("### Preview Uploaded Data", batch_df.head())

    required_cols = [
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    if all(col in batch_df.columns for col in required_cols):
      if st.button("Run Batch Prediction"):
        type_map = {
            "PAYMENT": 0,
            "TRANSFER": 1,
            "CASH_OUT": 2,
            "DEBIT": 3,
            "CASH_IN": 4,
        }
        batch_df["type_num"] = batch_df["type"].map(type_map)

        X_batch = batch_df[[
            "type_num",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ]]
        predictions = model.predict(X_batch)

        batch_df["Fraud_Prediction"] = np.where(
            predictions == 1, "FRAUD", "LEGITIMATE"
        )

        st.success("Analysis Complete!")
        st.dataframe(batch_df)

        csv = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Classified Report",
            data=csv,
            file_name="fraud_analysis_results.csv",
            mime="text/csv",
        )
    else:
      st.error(f"CSV must contain the following columns: {required_cols}")
