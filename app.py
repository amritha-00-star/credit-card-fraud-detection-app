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
    page_icon="🛡️️",
    layout="wide",
)

# Custom Styling: CreditVault Modern UI Theme (Mint Green & Dark Navy Slate)
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
        font-size: 32px !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #8B98A5 !important;
        font-size: 14px !important;
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


with st.spinner("Initializing Intelligence Engine..."):
  df, model = load_data_and_train()

# Sidebar Navigation
st.sidebar.title("🛡️ Fraud Engine")
page = st.sidebar.radio(
    "Navigate to:",
    ["📊 Executive Analytics", "🚨 Real-Time Predictor", "📁 Batch CSV Scanner"],
)

# PAGE 1: EXECUTIVE ANALYTICS
if page == "📊 Executive Analytics":

  # HERO BANNER SECTION (Matches Screenshot 1)
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

  # DATASET METRICS SECTION (Matches Screenshot 2)
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

  # CLASS DISTRIBUTION DONUT CHART (Matches Screenshot 3)
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

  # MODEL INSIGHTS CARDS (Matches Screenshot 4 & 5)
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
