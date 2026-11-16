
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def evaluate_model(model_name, y_test, y_pred, y_proba):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    precision_vals, recall_vals, _ = precision_recall_curve(
        y_test, y_proba
    )
    pr_auc = auc(recall_vals, precision_vals)

    results = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "PR-AUC": pr_auc
    }

    return results

if __name__ == "__main__":
    y_test = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 0, 0, 1]
    y_proba = [0.10, 0.85, 0.20, 0.40, 0.90]

    results = evaluate_model(
        "Test Model",
        y_test,
        y_pred,
        y_proba
    )

    print("\nModel Evaluation Results:")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
