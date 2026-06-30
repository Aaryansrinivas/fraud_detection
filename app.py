import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# ---------------------------
# LOAD MODEL
# ---------------------------

model = joblib.load("fraud_model.pkl")

# ---------------------------
# TITLE
# ---------------------------

st.title("💳 Credit Card Fraud Detection System")

st.markdown(
    """
    Upload transaction data and detect potentially fraudulent transactions.
    """
)

# ---------------------------
# THRESHOLD
# ---------------------------

threshold = st.sidebar.slider(
    "Fraud Detection Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

st.sidebar.write(
    f"Current Threshold: {threshold}"
)

# ---------------------------
# FILE UPLOAD
# ---------------------------

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ---------------------------
# RISK FUNCTION
# ---------------------------

def get_risk_level(score):

    if score < 30:
        return "Low"

    elif score < 70:
        return "Medium"

    else:
        return "High"

# ---------------------------
# PREDICTION
# ---------------------------

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    # Save actual labels if present
    if "Class" in data.columns:
        actual_labels = data["Class"]
        data = data.drop("Class", axis=1)

    st.subheader("Uploaded Data")

    st.dataframe(data.head())

    probability = model.predict_proba(data)[:, 1]

    prediction = (
        probability >= threshold
    ).astype(int)

    data["Fraud_Probability"] = (
        probability * 100
    ).round(2)

    data["Risk_Score"] = (
        probability * 100
    ).round(2)

    data["Risk_Level"] = data[
        "Risk_Score"
    ].apply(get_risk_level)

    data["Prediction"] = np.where(
        prediction == 1,
        "Fraud",
        "Genuine"
    )

    # ---------------------------
    # DASHBOARD METRICS
    # ---------------------------

    total_transactions = len(data)

    fraud_count = (
        prediction == 1
    ).sum()

    genuine_count = (
        prediction == 0
    ).sum()

    avg_risk = data[
        "Risk_Score"
    ].mean()

    st.subheader("Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        total_transactions
    )

    col2.metric(
        "Frauds Detected",
        fraud_count
    )

    col3.metric(
        "Genuine Transactions",
        genuine_count
    )

    col4.metric(
        "Average Risk Score",
        round(avg_risk, 2)
    )

    # ---------------------------
    # FRAUD DISTRIBUTION
    # ---------------------------

    st.subheader("Fraud Distribution")

    fraud_chart = pd.DataFrame(
        {
            "Count": [
                genuine_count,
                fraud_count
            ]
        },
        index=[
            "Genuine",
            "Fraud"
        ]
    )

    st.bar_chart(fraud_chart)

    # ---------------------------
    # RISK LEVEL DISTRIBUTION
    # ---------------------------

    st.subheader("Risk Level Distribution")

    risk_counts = data[
        "Risk_Level"
    ].value_counts()

    st.bar_chart(risk_counts)

    # ---------------------------
    # RISK SCORE HISTOGRAM
    # ---------------------------

    st.subheader("Risk Score Distribution")

    st.area_chart(
        data["Risk_Score"]
    )

    # ---------------------------
    # TOP 10 SUSPICIOUS
    # ---------------------------

    st.subheader(
        "Top 10 Most Suspicious Transactions"
    )

    top10 = data.sort_values(
        by="Risk_Score",
        ascending=False
    ).head(10)

    st.dataframe(top10)

    # ---------------------------
    # HIGH RISK TRANSACTIONS
    # ---------------------------

    st.subheader(
        "High Risk Transactions"
    )

    high_risk = data[
        data["Risk_Score"] >= 70
    ]

    st.dataframe(high_risk)

    # ---------------------------
    # FULL RESULTS
    # ---------------------------

    st.subheader("Prediction Results")

    st.dataframe(data)

    # ---------------------------
    # DOWNLOAD
    # ---------------------------

    csv = data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Results",
        data=csv,
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )

else:

    st.info(
        "Please upload a CSV file."
    )
