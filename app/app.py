import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('../models/fraud_model_final.pkl')
scaler = joblib.load('../models/scaler_final.pkl')

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection System")
st.markdown("Upload transaction data (CSV, Excel, TXT) to detect fraud.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "txt"]
)

def predict(df, threshold=0.9):

    # 🔥 FIX 1: remove target column if it exists
    df = df.drop(columns=['Class'], errors='ignore')

    # 🔥 FIX 2: ensure correct feature order (VERY IMPORTANT)
    if hasattr(model, "feature_names_in_"):
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    # scale
    df_scaled = scaler.transform(df)

    # predict probabilities
    probs = model.predict_proba(df_scaled)[:, 1]
    preds = (probs >= threshold).astype(int)

    df = df.copy()
    df['Fraud_Probability'] = probs
    df['Prediction'] = preds

    return df

if uploaded_file is not None:

    # Detect file type and read
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            df = pd.read_csv(uploaded_file, delimiter='\t')
        else:
            st.error("Unsupported file format")
            st.stop()

        st.success("✅ File uploaded successfully!")

        # Show data preview
        st.subheader("📊 Data Preview")
        st.dataframe(df.head())

        # Predict button
        if st.button("🚀 Run Fraud Detection"):
            result_df = predict(df)

            st.subheader("🔍 Prediction Results")
            st.dataframe(result_df.head())

            # Summary
            fraud_count = result_df['Prediction'].sum()
            total = len(result_df)

            st.subheader("📈 Summary")
            st.write(f"Total Transactions: {total}")
            st.write(f"Fraudulent Transactions: {fraud_count}")

            # Download results
            csv = result_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                "⬇️ Download Results",
                csv,
                "fraud_predictions.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")