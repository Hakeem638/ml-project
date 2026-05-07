import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("../models/fraud_model_final.pkl")
scaler = joblib.load("../models/scaler_final.pkl")


def predict_transactions(df, threshold=0.9):

    # Remove target column if uploaded
    df = df.drop(columns=["Class"], errors="ignore")

    # Ensure correct feature order
    if hasattr(model, "feature_names_in_"):
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Scale data
    scaled_data = scaler.transform(df)

    # Predict probabilities
    probs = model.predict_proba(scaled_data)[:, 1]

    # Apply threshold
    preds = (probs >= threshold).astype(int)

    # Create result dataframe
    result_df = df.copy()

    result_df["Fraud_Probability"] = probs
    result_df["Prediction"] = preds

    return result_df
