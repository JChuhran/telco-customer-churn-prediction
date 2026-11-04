"""Telco Customer Churn — advanced model tuning.

Milestone 3 work for IT4793.
Author: Alex Clark

Trains Random Forest and XGBoost on the cleaned Telco data and reports
F1 and PR-AUC on the churn class, the project success metrics.

Put this file in the same folder as WA_Fn-UseC_-Telco-Customer-Churn.csv
and run:
    python advanced_tuning.py
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

DATA = Path(__file__).with_name("WA_Fn-UseC_-Telco-Customer-Churn.csv")


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["customerID"])
    return df


def score(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, proba)
    print(f"\n===== {name} =====")
    print(f"F1-Score (Churn=1) : {f1_score(y_test, pred):.4f}")
    print(f"PR-AUC             : {auc(recall, precision):.4f}")
    print(f"ROC-AUC            : {roc_auc_score(y_test, proba):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, digits=4))


def main():
    df = load_and_clean(DATA)
    print(f"clean shape: {df.shape}")
    print(f"churn rate: {df['Churn'].mean():.1%}")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64", "int32"]).columns.tolist()

    # Trees do not need scaling. One-hot the text columns only.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # class_weight / scale_pos_weight handle the 26.6% churn imbalance.
    forest = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    forest.fit(X_train, y_train)
    score("Random Forest", forest, X_test, y_test)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    boost = Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=neg / pos,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    boost.fit(X_train, y_train)
    score("XGBoost", boost, X_test, y_test)


if __name__ == "__main__":
    main()
