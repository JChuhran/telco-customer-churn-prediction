"""
Telco Customer Churn — Baseline Model Pipeline
Milestone 2 Deliverable for IT4793

Builds a baseline Logistic Regression model based on eda_cleaning.py
and feature_engineering.py pipelines.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    auc,
    roc_auc_score
)

# Import existing cleaning and feature engineering functions
from eda_cleaning import load_and_clean
from feature_engineering import engineer_features

# Path definition
DATA_PATH = Path(__file__).with_name("WA_Fn-UseC_-Telco-Customer-Churn.csv")

def build_and_evaluate_baseline():
    # 1. Load, clean, and engineer features
    df = load_and_clean(DATA_PATH)
    df = engineer_features(df)

    # 2. Separate Features and Target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 3. Identify feature types
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # 4. Define Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
        ]
    )

    # 5. Define Model Pipeline
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    # 6. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 7. Fit Model
    model_pipeline.fit(X_train, y_train)

    # 8. Predictions & Probability Scores
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    # 9. Compute Metrics
    f1 = f1_score(y_test, y_pred)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_vals, precision_vals)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n=================== BASELINE MODEL EVALUATION ===================")
    print(f"F1-Score (Churn=1) : {f1:.4f}")
    print(f"PR-AUC             : {pr_auc:.4f}")
    print(f"ROC-AUC            : {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Extract Top Features
    ohe_feature_names = model_pipeline.named_steps["preprocessor"] \
                                      .named_transformers_["cat"] \
                                      .get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(ohe_feature_names)
    coefficients = model_pipeline.named_steps["classifier"].coef_[0]

    coef_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Coefficient": coefficients
    }).sort_values(by="Coefficient", key=abs, ascending=False)

    print("\nTop 10 Influential Features:")
    print(coef_df.head(10).to_string(index=False))

if __name__ == "__main__":
    build_and_evaluate_baseline()