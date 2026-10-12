"""Telco Customer Churn — cleaning and exploratory data analysis.

Milestone 2 work for IT4793.
Author: Alex Clark
Dataset: Kaggle Telco Customer Churn (BlastChar), file
WA_Fn-UseC_-Telco-Customer-Churn.csv

Run from the folder that contains the CSV:
    python eda_cleaning.py
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).with_name("WA_Fn-UseC_-Telco-Customer-Churn.csv")


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"raw shape: {df.shape}")

    # TotalCharges is stored as text. Tenure-0 accounts have a blank bill.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank = df["TotalCharges"].isna().sum()
    df = df.dropna(subset=["TotalCharges"]).copy()

    # Target as 0/1. customerID is an identifier, not a feature.
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["customerID"])

    print(f"dropped blank TotalCharges: {blank}")
    print(f"clean shape: {df.shape}")
    print(f"churn rate: {df['Churn'].mean():.1%}")
    return df


def eda(df: pd.DataFrame) -> None:
    print("\nChurn rate by contract")
    print(df.groupby("Contract")["Churn"].mean().round(4))

    print("\nChurn rate by internet service")
    print(df.groupby("InternetService")["Churn"].mean().round(4))

    print("\nChurn rate by tech support")
    print(df.groupby("TechSupport")["Churn"].mean().round(4))

    print("\nFindings")
    print("- Month-to-month contracts churn at ~42.7%; two-year contracts at ~2.8%.")
    print("- Fiber optic churns at ~41.9%; DSL at ~19.0%; no internet at ~7.4%.")
    print("- Customers with no tech support churn at ~41.6%; with support, ~15.2%.")
    print("- The high-risk profile is month-to-month, fiber, and no support add-on.")


if __name__ == "__main__":
    frame = load_and_clean(DATA)
    eda(frame)
