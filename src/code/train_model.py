import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve


def load_data(path):
    df = pd.read_csv(path)

    return df


def preprocess_data(df):

    X = df.drop("Class", axis=1)

    y = df["Class"]

    return X, y


def split_data(X, y):

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def scale_data(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train):

    model = LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced")

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test, threshold=0.9):

    y_probs = model.predict_proba(X_test)[:, 1]

    y_pred = (y_probs >= threshold).astype(int)

    print(classification_report(y_test, y_pred))

    print(confusion_matrix(y_test, y_pred))


def save_model(model, scaler):

    joblib.dump(model, "/models/fraud_model.pkl")

    joblib.dump(scaler, "/models/scaler.pkl")

    print("Model and scaler saved successfully!")


def main():

    print("Loading data...")
    df = load_data("../data/creditcard.csv")

    print("Preprocessing data...")
    X, y = preprocess_data(df)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    print("Training model...")
    model = train_model(X_train_scaled, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test_scaled, y_test)

    print("Saving model...")
    save_model(model, scaler)


if __name__ == "__main__":
    main()
