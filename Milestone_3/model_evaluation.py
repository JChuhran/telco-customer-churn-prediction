
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc
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


def load_model_results():
    file_path = Path(__file__).parent / "outputs" / "model_comparison.csv"

    df = pd.read_csv(file_path)

    print(f"\nSuccessfully loaded {len(df)} model evaluation records.")

    return df




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


def compare_models(df):
    best_f1 = df[df["Operating point"] == "Best-F1 threshold"]

    columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "PR-AUC"
    ]

    comparison = best_f1[columns].copy()
    comparison = comparison.sort_values(by="F1", ascending=False)

    print("\nModel Comparison (Best-F1 Threshold):")
    print(comparison.to_string(index=False, float_format="%.4f"))

    return comparison



def identify_best_models(comparison):
    metrics = ["F1", "PR-AUC", "Recall"]

    print("\nBest Performing Models:")

    for metric in metrics:
        best_row = comparison.loc[comparison[metric].idxmax()]

        print(
            f"Best {metric}: {best_row['Model']} "
            f"({best_row[metric]:.4f})"
        )

def analyze_errors(df):
    best_f1 = df[df["Operating point"] == "Best-F1 threshold"]

    columns = ["Model", "Missed churners", "False alarms"]
    errors = best_f1[columns].copy()

    errors = errors.sort_values(by="Missed churners")

    print("\nModel Error Analysis (Best-F1 Threshold):")
    print(errors.to_string(index=False))

    return errors

def plot_model_errors(errors):
    chart_data = errors.set_index("Model")

    ax = chart_data.plot(
        kind="bar",
        figsize=(10, 6),
        width=0.75
    )

    ax.set_title("Model Prediction Errors (Best-F1 Threshold)")
    ax.set_xlabel("Machine Learning Model")
    ax.set_ylabel("Number of Customers")

    plt.xticks(rotation=25, ha="right")

    plt.tight_layout()

    output_path = Path(__file__).parent / "outputs" / "evaluation_errors.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"\nError comparison graph saved to: {output_path}")

    plt.show()

def generate_evaluation_summary(comparison, errors):
    best_f1 = comparison.loc[comparison["F1"].idxmax()]
    best_pr_auc = comparison.loc[comparison["PR-AUC"].idxmax()]
    best_recall = comparison.loc[comparison["Recall"].idxmax()]

    fewest_missed = errors.loc[errors["Missed churners"].idxmin()]
    fewest_alarms = errors.loc[errors["False alarms"].idxmin()]

    print("\n========== MODEL EVALUATION SUMMARY ==========")

    print(
        f"Highest F1-Score: {best_f1['Model']} "
        f"({best_f1['F1']:.4f})"
    )

    print(
        f"Highest PR-AUC: {best_pr_auc['Model']} "
        f"({best_pr_auc['PR-AUC']:.4f})"
    )

    print(
        f"Highest Recall: {best_recall['Model']} "
        f"({best_recall['Recall']:.4f})"
    )

    print(
        f"Fewest Missed Churners: {fewest_missed['Model']} "
        f"({fewest_missed['Missed churners']})"
    )

    print(
        f"Fewest False Alarms: {fewest_alarms['Model']} "
        f"({fewest_alarms['False alarms']})"
    )

    print("\nBusiness Interpretation:")
    print(
        "Models with higher recall identify more customers at risk "
        "of leaving, potentially allowing the company to intervene."
    )
    print(
        "However, higher recall can also result in more false alarms, "
        "which may increase the cost of unnecessary retention offers."
    )
    print(
        "The best model depends on whether the company prioritizes "
        "catching more churners or minimizing unnecessary interventions."
    )

def save_evaluation_summary(comparison, errors):
    best_f1 = comparison.loc[comparison["F1"].idxmax()]
    best_pr_auc = comparison.loc[comparison["PR-AUC"].idxmax()]
    best_recall = comparison.loc[comparison["Recall"].idxmax()]

    fewest_missed = errors.loc[errors["Missed churners"].idxmin()]
    fewest_alarms = errors.loc[errors["False alarms"].idxmin()]

    summary = (
        "TELCO CUSTOMER CHURN - MODEL EVALUATION SUMMARY\n"
        "=============================================\n\n"
        "Results at the Best-F1 threshold:\n\n"
        f"Highest F1-Score: {best_f1['Model']} ({best_f1['F1']:.4f})\n"
        f"Highest PR-AUC: {best_pr_auc['Model']} ({best_pr_auc['PR-AUC']:.4f})\n"
        f"Highest Recall: {best_recall['Model']} ({best_recall['Recall']:.4f})\n"
        f"Fewest Missed Churners: {fewest_missed['Model']} "
        f"({fewest_missed['Missed churners']})\n"
        f"Fewest False Alarms: {fewest_alarms['Model']} "
        f"({fewest_alarms['False alarms']})\n\n"
        "BUSINESS INTERPRETATION\n"
        "-----------------------\n"
        "Higher recall helps identify more customers at risk of leaving, "
        "but may also increase false alarms and unnecessary retention costs.\n"
        "The best model depends on the company's priorities and the cost "
        "of missed churners compared with unnecessary interventions.\n"

        "\nTHRESHOLD ANALYSIS\n"
        "------------------\n"
        "Random Forest achieved an F1-score of 0.6219 at its "
        "Best-F1 threshold and 0.6341 at its recall-target threshold.\n"
        "The recall-target threshold reduced false alarms from "
        "262 to 214 but increased missed churners from 87 to 101.\n"
        "This demonstrates the trade-off between identifying "
        "customers at risk and avoiding unnecessary interventions.\n\n"
        "CONCLUSION\n"
        "----------\n"
        "Random Forest is a promising candidate because of its "
        "competitive F1-score and flexible threshold performance.\n"
        "XGBoost achieved the highest PR-AUC, while LightGBM "
        "identified the largest proportion of actual churners "
        "at the Best-F1 operating point.\n"
        "Final model selection should consider business costs "
        "and performance on future unseen data.\n"

    )

    output_path = Path(__file__).parent / "outputs" / "evaluation_summary.txt"
    output_path.write_text(summary, encoding="utf-8")

    print(f"\nEvaluation summary saved to: {output_path}")


def compare_thresholds(df):
    columns = [
        "Model",
        "Operating point",
        "Threshold",
        "Precision",
        "Recall",
        "F1",
        "Missed churners",
        "False alarms"
    ]

    threshold_comparison = df[columns].copy()

    threshold_comparison = threshold_comparison.sort_values(
        by=["Model", "F1"],
        ascending=[True, False]
    )

    print("\n========== THRESHOLD COMPARISON ==========")
    print(
        threshold_comparison.to_string(
            index=False,
            float_format="%.4f"
        )
    )

    return threshold_comparison


if __name__ == "__main__":
    df = load_model_results()
    comparison = compare_models(df)
    identify_best_models(comparison)

    errors = analyze_errors(df)
    plot_model_errors(errors)

    generate_evaluation_summary(comparison, errors)
    save_evaluation_summary(comparison, errors)

    compare_thresholds(df)







