



from pathlib import Path
import pandas as pd

from eda_cleaning import load_and_clean


DATA = Path(__file__).parent / "Data" / "telco_customer_churn_kaggle_dataset_raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def engineer_features(df):
    df = df.copy()

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=["New", "Short-term", "Established", "Long-term"]
    )

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["NumServices"] = (
        df[service_columns] == "Yes"
    ).sum(axis=1)

    df["NumServices"] += (
        df["InternetService"] != "No"
    ).astype(int)

    df["MonthlyChargePerService"] = (
        df["MonthlyCharges"] / df["NumServices"].clip(lower=1)
    )

    return df

if __name__ == "__main__":
    df = load_and_clean(DATA)
    df = engineer_features(df)

    print("\nEngineered Features:")
    print(df[[
        "tenure",
        "TenureGroup",
        "NumServices",
        "MonthlyChargePerService"
    ]].head(10))

    print("\nDataset Shape:")
    print(df.shape)

    print("\nMissing Values:")
    print(df[[
        "TenureGroup",
        "NumServices",
        "MonthlyChargePerService"
    ]].isnull().sum())

