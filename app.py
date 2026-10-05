import glob
import os
import kagglehub
import numpy as np
import pandas as pd
import plotly.express as px
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

# Page Configuration
st.set_page_config(
    page_title="Enterprise Fraud Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# Custom Styling: CreditVault Theme (Mint Green & Dark Navy Slate)
st.markdown(
    """
    <style>
    /* Global App Background */
    .stApp {
        background-color: #0B0E17 !important;
        color: #E6EDF3 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar Custom Styling */
    section[data-testid="stSidebar"] {
        background-color: #121721 !important;
        border-right: 1px solid #1F2937 !important;
    }

    /* Radio Buttons & Menu Styling in Sidebar */
    div[data-testid="stSidebar"] label {
        color: #8B98A5 !important;
        font-size: 15px !important;
        font-weight: 500 !important;
    }
    
    div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        color: #50E3C2 !important;
    }
    
    /* Active Selected Radio Item */
    div[data-testid="stSidebar"] div[role="radiogroup"] [aria-checked="true"] {
        background-color: rgba(80, 227, 194, 0.1) !important;
        border-left: 4px solid #50E3C2 !important;
        border-radius: 4px;
        padding-left: 8px;
    }

    div[data-testid="stSidebar"] div[role="radiogroup"] [aria-checked="true"] p {
        color: #50E3C2 !important;
        font-weight: 700 !important;
    }
    
    /* Typography Overrides */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
    }
    
    .mint-text {
        color: #50E3C2 !important;
    }

    /* Modern Card Containers */
    .hero-card {
        background: linear-gradient(135deg, #131B2A 0%, #161F33 100%);
        border: 1px solid #233044;
        border-radius: 16px;
        padding: 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }

    .info-card {
        background-color: #161F30;
        border: 1px solid #233044;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Tag / Badge Pills */
    .badge-pill {
        display: inline-block;
        background-color: #1C273A;
        border: 1px solid #2A3B58;
        color: #8B98A5;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    /* Metric Cards Styling */
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 30px !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #8B98A5 !important;
        font-size: 13px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Primary Buttons */
    .stButton>button {
        background-color: #50E3C2 !important;
        color: #0B0E17 !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 12px 24px !important;
        width: 100%;
        transition: all 0.2s ease-in-out;
    }
    
    .stButton>button:hover {
        background-color: #38C7A7 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(80, 227, 194, 0.3);
    }

    /* Input Controls */
    input, select, div[data-baseweb="select"] {
        background-color: #121824 !important;
        color: #FFFFFF !important;
        border: 1px solid #233044 !important;
        border-radius: 8px !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load Dataset, Train Model & Compute Evaluation Metrics
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

  # Train / Test Split for Model Evaluation
  X_train, X_test, y_train, y_test = train_test_split(
      X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
  )

  model = LogisticRegression(max_iter=1000)
  model.fit(X_train, y_train)

  # Evaluation Scores
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


with st.spinner("Initializing Intelligence Engine & Evaluation Suite..."):
  df, model, metrics = load_data_and_train()

# Sidebar Navigation
st.sidebar.markdown(
    "<h3 style='color:#50E3C2 !important; margin-bottom: 20px;'>🛡️ Fraud"
    " Engine Menu</h3>",
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Navigation Options:",
    [
        "📊 Executive Analytics",
        "🚨 Real-Time Predictor",
        "📁 Batch CSV Scanner",
        "📈 Model Performance",
    ],
)

# PAGE 1: EXECUTIVE ANALYTICS
if page == "📊 Executive Analytics":

  st.markdown(
      """
        <div class="hero-card">
            <div style="font-size: 12px; font-weight: 700; color: #50E3C2; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px;">
                🟢 Credit Card Fraud Detection
            </div>
            <h1 style="font-size: 40px; margin-bottom: 8px; line-height: 1.2;">
                Fraud signals.<br><span class="mint-text">Brought into focus.</span>
            </h1>
            <p style="color: #8B98A5; font-size: 16px; margin-bottom: 20px;">
                Go beyond the spreadsheet. Explore transaction patterns, understand the model, and make every prediction clear.
            </p>
            <div>
                <span class="badge-pill">6 Core Features</span>
                <span class="badge-pill">Logistic Regression</span>
                <span class="badge-pill">Kaggle PaySim Engine</span>
            </div>
        </div>
    """,
      unsafe_allow_html=True,
  )

  st.markdown(
      "<div style='font-size: 12px; color: #8B98A5; text-transform: uppercase; font-weight: 600;'>01 / THE DATASET</div>",
      unsafe_allow_html=True,
  )
  st.markdown(
      "<h2 style='margin-bottom: 20px;'>The full picture, at a glance.</h2>",
      unsafe_allow_html=True,
  )

  total_tx = len(df)
  total_fraud = int(df["isFraud"].sum())
  legit_count = total_tx - total_fraud
  fraud_rate = (total_fraud / total_tx) * 100

  m_col1, m_col2, m_col3, m_col4 = st.columns(4)

  with m_col1:
    st.markdown(
        f'<div class="info-card"><div'
        ' style="color:#8B98A5;font-size:12px;">Total'
        ' transactions</div><div style="font-size:26px; font-weight:700;'
        ' color:#FFF;">'
        f"{total_tx:,}"
        "</div></div>",
        unsafe_allow_html=True,
    )
  with m_col2:
    st.markdown(
        f'<div class="info-card"><div'
        ' style="color:#8B98A5;font-size:12px;">Legitimate</div><div'
        ' style="font-size:26px; font-weight:700; color:#FFF;">'
        f"{legit_count:,}"
        "</div></div>",
        unsafe_allow_html=True,
    )
  with m_col3:
    st.markdown(
        f'<div class="info-card"><div'
        ' style="color:#8B98A5;font-size:12px;">Fraudulent</div><div'
        ' style="font-size:26px; font-weight:700; color:#FFF;">'
        f"{total_fraud:,}"
        "</div></div>",
        unsafe_allow_html=True,
    )
  with m_col4:
    st.markdown(
        f'<div class="info-card"><div'
        ' style="color:#8B98A5;font-size:12px;">Fraud rate</div><div'
        ' style="font-size:26px; font-weight:700; color:#50E3C2;">'
        f"{fraud_rate:.3f}%"
        "</div></div>",
        unsafe_allow_html=True,
    )

  st.markdown("<br>", unsafe_allow_html=True)

  chart_col1, chart_col2 = st.columns(2)

  with chart_col1:
    st.subheader("Class Distribution")
    status_df = pd.DataFrame({
        "Status": ["Legitimate", "Fraudulent"],
        "Count": [legit_count, total_fraud],
    })
    fig_donut = px.pie(
        status_df,
        values="Count",
        names="Status",
        hole=0.65,
        color_discrete_sequence=["#50E3C2", "#FF4D4D"],
        template="plotly_dark",
    )
    fig_donut.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        annotations=[{
            "text": f"<b>{fraud_rate:.3f}%</b><br><span"
            ' style="font-size:12px;color:#8B98A5;">FRAUD RATE</span>',
            "x": 0.5,
            "y": 0.5,
            "font_size": 22,
            "showarrow": False,
        }],
    )
    st.plotly_chart(fig_donut, use_container_width=True)

  with chart_col2:
    st.subheader("Fraud Count by Type")
    fraud_type = df[df["isFraud"] == 1]["type"].value_counts().reset_index()
    fraud_type.columns = ["Transaction Type", "Fraud Count"]
    fig_bar = px.bar(
        fraud_type,
        x="Transaction Type",
        y="Fraud Count",
        text_auto=True,
        template="plotly_dark",
    )
    fig_bar.update_traces(marker_color="#50E3C2")
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

  st.markdown("<br>", unsafe_allow_html=True)
  st.markdown(
      "<div style='font-size: 12px; color: #8B98A5; text-transform: uppercase; font-weight: 600;'>02 / ARCHITECTURE</div>",
      unsafe_allow_html=True,
  )
  st.markdown(
      "<h2>Inside the classification model.</h2>", unsafe_allow_html=True
  )

  info1, info2 = st.columns(2)
  with info1:
    st.markdown(
        """
            <div class="info-card">
                <span class="badge-pill">THE INPUT</span>
                <h3 style="margin-top:10px;">6 Core Dimensions</h3>
                <p style="color:#8B98A5; font-size:14px;">
                    Transaction type, amount, sender balances, and recipient balances processed via numerical encoding.
                </p>
            </div>
        """,
        unsafe_allow_html=True,
    )

  with info2:
    st.markdown(
        """
            <div class="info-card">
                <span class="badge-pill">THE APPROACH</span>
                <h3 style="margin-top:10px;">Logistic Regression Classifier</h3>
                <p style="color:#8B98A5; font-size:14px;">
                    Trained with random under-sampling on balanced transaction distributions for robust probability scoring.
                </p>
            </div>
        """,
        unsafe_allow_html=True,
    )

# PAGE 2: REAL-TIME PREDICTOR
elif page == "🚨 Real-Time Predictor":
  st.markdown("<h2>Score a Transaction</h2>", unsafe_allow_html=True)
  st.caption("Submit custom parameters to compute real-time fraud scores.")

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

  if st.button("Evaluate Fraud Probability"):
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

    st.markdown("<br>", unsafe_allow_html=True)
    res_c1, res_c2 = st.columns(2)

    with res_c1:
      if prediction == 1:
        st.error("🚨 **Classification:** FRAUD DETECTED")
      else:
        st.success("✅ **Classification:** LEGITIMATE TRANSACTION")

    with res_c2:
      st.metric("Fraud Probability Score", f"{prob * 100:.2f}%")

# PAGE 3: BATCH CSV SCANNER
elif page == "📁 Batch CSV Scanner":
  st.markdown("<h2>Bulk Fraud File Analysis</h2>", unsafe_allow_html=True)
  st.caption("Upload a CSV file of transactions for batch scoring.")

  uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

  if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.write("### Preview Data", batch_df.head())

    required_cols = [
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    if all(col in batch_df.columns for col in required_cols):
      if st.button("Run Batch Scoring"):
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

# PAGE 4: MODEL PERFORMANCE & EVALUATION METRICS
elif page == "📈 Model Performance":
  st.markdown(
      "<div style='font-size: 12px; color: #8B98A5; text-transform: uppercase; font-weight: 600;'>03 / EVALUATION METRICS</div>",
      unsafe_allow_html=True,
  )
  st.markdown(
      "<h2>Model Performance & Diagnostic Suite</h2>", unsafe_allow_html=True
  )
  st.caption(
      "Detailed classification metrics, confusion matrix, and ROC-AUC curve computed on holdout test set."
  )

  st.markdown("<br>", unsafe_allow_html=True)

  # Key Metric Cards
  kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
  kpi1.metric("Accuracy Score", f"{metrics['accuracy']*100:.2f}%")
  kpi2.metric("Precision", f"{metrics['precision']*100:.2f}%")
  kpi3.metric("Recall (Sensitivity)", f"{metrics['recall']*100:.2f}%")
  kpi4.metric("F1 Score", f"{metrics['f1']*100:.2f}%")
  kpi5.metric("ROC AUC Value", f"{metrics['roc_auc']:.4f}")

  st.markdown("<br>", unsafe_allow_html=True)

  col_roc, col_cm = st.columns(2)

  with col_roc:
    st.subheader("ROC-AUC Curve")
    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(
            x=metrics["fpr"],
            y=metrics["tpr"],
            mode="lines",
            name=f"Logistic Regression (AUC = {metrics['roc_auc']:.3f})",
            line=dict(color="#50E3C2", width=3),
        )
    )
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Baseline",
            line=dict(color="#FF4D4D", dash="dash"),
        )
    )
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.4, y=0.1),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

  with col_cm:
    st.subheader("Confusion Matrix")
    z = metrics["cm"]
    x_labels = ["Predicted Legit", "Predicted Fraud"]
    y_labels = ["Actual Legit", "Actual Fraud"]

    # Replaced figure_factory with plotly express (px.imshow) to avoid scipy dependency
    fig_cm = px.imshow(
        z,
        x=x_labels,
        y=y_labels,
        text_auto=True,
        color_continuous_scale=[[0, "#161F30"], [1, "#50E3C2"]],
        aspect="auto",
    )
    fig_cm.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

  # Model Specifications & Feature Coefficients
  st.markdown("<br>", unsafe_allow_html=True)
  st.markdown("<h3>Model Parameters & Feature Weights</h3>", unsafe_allow_html=True)

  spec_col1, spec_col2 = st.columns(2)

  with spec_col1:
    st.markdown(
        """
            <div class="info-card">
                <span class="badge-pill">SPECIFICATIONS</span>
                <ul style="color:#8B98A5; font-size:14px; margin-top:10px; line-height:1.8;">
                    <li><b>Algorithm:</b> Logistic Regression (Scikit-Learn)</li>
                    <li><b>Max Iterations:</b> 1000</li>
                    <li><b>Resampling Method:</b> Random Under-Sampling</li>
                    <li><b>Test Split:</b> 30% Holdout Test Dataset</li>
                </ul>
            </div>
        """,
        unsafe_allow_html=True,
    )

  with spec_col2:
    coefficients = model.coef_[0]
    coef_df = pd.DataFrame(
        {"Feature": metrics["feature_cols"], "Coefficient": coefficients}
    ).sort_values(by="Coefficient", ascending=True)

    fig_coef = px.bar(
        coef_df,
        x="Coefficient",
        y="Feature",
        orientation="h",
        title="Feature Coefficients (Impact on Fraud Probability)",
        template="plotly_dark",
    )
    fig_coef.update_traces(marker_color="#50E3C2")
    fig_coef.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_coef, use_container_width=True)
