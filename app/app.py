import streamlit as st
import pandas as pd
import sys
import os

# allow app to access src folder
sys.path.append(os.path.abspath(".."))

from src.code.predict import predict_transactions

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Fraud Detection System", layout="wide")

# ==============================
# TITLE
# ==============================
st.title("💳 Credit Card Fraud Detection System")

st.markdown(
    """
    Upload transaction datasets and detect potentially fraudulent transactions
    using a trained Machine Learning model.
    """
)

# ==============================
# FILE UPLOADER
# ==============================
uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "txt"])

# ==============================
# PROCESS FILE
# ==============================
if uploaded_file is not None:
    try:
        # ==============================
        # READ FILE
        # ==============================
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        elif uploaded_file.name.endswith(".txt"):
            df = pd.read_csv(uploaded_file, delimiter="\t")

        else:
            st.error("Unsupported file format")
            st.stop()

        st.success("✅ File uploaded successfully!")

        # ==============================
        # DATA PREVIEW
        # ==============================
        st.subheader("📊 Dataset Preview")

        st.dataframe(df.head())

        # ==============================
        # RUN PREDICTION
        # ==============================
        if st.button("🚀 Run Fraud Detection"):
            with st.spinner("Analyzing transactions..."):
                result_df = predict_transactions(df)

            st.success("Fraud detection completed!")

            # ==============================
            # RESULTS
            # ==============================
            st.subheader("🔍 Prediction Results")

            st.dataframe(result_df.head())

            # ==============================
            # SUMMARY
            # ==============================
            fraud_count = result_df["Prediction"].sum()

            total_transactions = len(result_df)

            st.subheader("📈 Summary")

            col1, col2 = st.columns(2)

            col1.metric("Total Transactions", total_transactions)

            col2.metric("Fraudulent Transactions", fraud_count)

            # ==============================
            # DOWNLOAD RESULTS
            # ==============================
            csv = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Results",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
