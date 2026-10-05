import glob
import os
import kagglehub
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Fraud Detection Engine", page_icon="🛡️", layout="wide"
)


@st.cache_resource
def load_data_and_train():
  try:
    path = kagglehub.dataset_download("ealaxi/paysim1")
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
      st.error("No CSV file found in Kaggle download.")
      st.stop()
    df = pd.read_csv(csv_files[0])
  except Exception as e:
    st.error(f"Error downloading or loading dataset: {e}")
    st.stop()

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

  fraud = df[df.isFraud == 1]
  legit = df[df.isFraud == 0].sample(n=len(fraud), random_state=42)
  balanced_df = pd.concat([legit, fraud], axis=0)

  X = balanced_df[feature_cols]
  y = balanced_df["isFraud"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.3, random_state=42, stratify=y
  )

  model = LogisticRegression(max_iter=1000)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  y_prob = model.predict_proba(X_test)[:, 1]

  acc = accuracy_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred)
  rec = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  roc_auc = roc_auc_score(y_test, y_prob)
  cm = confusion_matrix(y_test, y_pred)
  fpr, tpr, _ = roc_curve(y_test, y_prob)

  eval_metrics = {
      "accuracy": acc,
      "precision": prec,
      "recall": rec,
      "f1": f1,
      "roc_auc": roc_auc,
      "cm": cm,
      "fpr": fpr,
      "tpr": tpr,
      "feature_cols": feature_cols,
  }

  return df, model, eval_metrics


with st.spinner("Loading Fraud Detection Engine..."):
  df, model, metrics = load_data_and_train()

st.sidebar.title("🛡️ Fraud Engine")
page = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Dashboard Overview",
        "🚨 Real-Time Predictor",
        "📁 Batch CSV Scanner",
        "📈 Model Performance",
    ],
)

if page == "📊 Dashboard Overview":
  st.title("📊 Dashboard Overview")

  total_tx = len(df)
  total_fraud = int(df["isFraud"].sum())
  legit_count = total_tx - total_fraud
  fraud_rate = (total_fraud / total_tx) * 100

  m1, m2, m3, m4 = st.columns(4)
  m1.metric("Total Transactions", f"{total_tx:,}")
  m2.metric("Legitimate", f"{legit_count:,}")
  m3.metric("Fraudulent", f"{total_fraud:,}")
  m4.metric("Fraud Rate", f"{fraud_rate:.3f}%")

  st.divider()

  col1, col2 = st.columns(2)
  with col1:
    fig_pie = px.pie(
        names=["Legitimate", "Fraudulent"],
        values=[legit_count, total_fraud],
        hole=0.5,
        title="Transaction Class Distribution",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

  with col2:
    fraud_type = df[df["isFraud"] == 1]["type"].value_counts().reset_index()
    fraud_type.columns = ["Transaction Type", "Fraud Count"]
    fig_bar = px.bar(
        fraud_type,
        x="Transaction Type",
        y="Fraud Count",
        title="Fraud Cases by Transaction Type",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

elif page == "🚨 Real-Time Predictor":
  st.title("🚨 Real-Time Fraud Predictor")
  st.caption("Enter transaction details below to get an instant prediction.")

  type_map = {
      "PAYMENT": 0,
      "TRANSFER": 1,
      "CASH_OUT": 2,
      "DEBIT": 3,
      "CASH_IN": 4,
  }

  col1, col2 = st.columns(2)
  with col1:
    tx_type = st.selectbox("Transaction Type", list(type_map.keys()))
    amount = st.number_input("Amount ($)", value=1000.0, step=100.0)
    old_org = st.number_input(
        "Sender Initial Balance ($)", value=10000.0, step=100.0
    )
  with col2:
    new_org = st.number_input(
        "Sender New Balance ($)", value=9000.0, step=100.0
    )
    old_dest = st.number_input(
        "Receiver Initial Balance ($)", value=0.0, step=100.0
    )
    new_dest = st.number_input(
        "Receiver New Balance ($)", value=0.0, step=100.0
    )

  if st.button("Analyze Transaction"):
    input_data = np.array(
        [[type_map[tx_type], amount, old_org, new_org, old_dest, new_dest]]
    )
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.divider()
    if pred == 1:
      st.error(f"🚨 **Result: FRAUD DETECTED** (Probability: {prob*100:.2f}%)")
    else:
      st.success(
          f"✅ **Result: LEGITIMATE TRANSACTION** (Risk Score: {prob*100:.2f}%)"
      )

elif page == "📁 Batch CSV Scanner":
  st.title("📁 Batch CSV Scanner")
  st.caption("Upload a CSV file containing transaction data.")

  uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

  if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", batch_df.head())

    required_cols = [
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    if all(col in batch_df.columns for col in required_cols):
      if st.button("Run Classification"):
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

        batch_df["Prediction"] = np.where(
            model.predict(X_batch) == 1, "FRAUD", "LEGITIMATE"
        )
        st.success("Analysis Complete!")
        st.dataframe(batch_df)
    else:
      st.error(f"File must contain the columns: {required_cols}")

elif page == "📈 Model Performance":
  st.title("📈 Model Performance & Evaluation")
  st.caption(
      "Metrics calculated on the 30% holdout test dataset using Logistic"
      " Regression."
  )

  st.divider()

  m1, m2, m3, m4, m5 = st.columns(5)
  m1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
  m2.metric("Precision", f"{metrics['precision']*100:.2f}%")
  m3.metric("Recall", f"{metrics['recall']*100:.2f}%")
  m4.metric("F1 Score", f"{metrics['f1']*100:.2f}%")
  m5.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")

  st.divider()

  col_roc, col_cm = st.columns(2)

  with col_roc:
    st.subheader("ROC-AUC Curve")
    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(
            x=metrics["fpr"],
            y=metrics["tpr"],
            mode="lines",
            name=f"ROC Curve (AUC = {metrics['roc_auc']:.3f})",
        )
    )
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Baseline",
            line=dict(dash="dash", color="red"),
        )
    )
    fig_roc.update_layout(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate"
    )
    st.plotly_chart(fig_roc, use_container_width=True)

  with col_cm:
    st.subheader("Confusion Matrix")
    fig_cm = px.imshow(
        metrics["cm"],
        x=["Predicted Legit", "Predicted Fraud"],
        y=["Actual Legit", "Actual Fraud"],
        text_auto=True,
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_cm, use_container_width=True)
