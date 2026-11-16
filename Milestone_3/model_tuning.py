"""Telco Customer Churn: full advanced model tuning.

Milestone 3 work for IT4793.

What this script does
---------------------
1. Loads and cleans the data with eda_cleaning.load_and_clean and adds the
   engineered features from feature_engineering.engineer_features (both live
   in Milestone_2/), so every model sees the same inputs as the baseline.
2. Uses the same 80/20 stratified split (random_state=42) as the baseline.
3. Tunes Logistic Regression, Random Forest, XGBoost and LightGBM with a
   randomized search and stratified cross-validation on the training set only.
   The search scores PR-AUC (average precision), a project success metric, and
   also chooses the imbalance strategy (class weighting, SMOTE, or both).
4. Picks a decision threshold from out-of-fold training predictions (never
   from the test set) at two operating points: best F1, and highest threshold
   that still reaches the recall target (default 0.70).
5. Evaluates everything once on the held-out test set and saves a comparison
   table, best parameters, a markdown summary and plots to Milestone_3/outputs/.

Setup and run (from anywhere inside the repo):
    pip install pandas scikit-learn xgboost lightgbm imbalanced-learn matplotlib scipy
    python Milestone_3/model_tuning.py              # full run
    python Milestone_3/model_tuning.py --quick      # fast smoke test

The raw CSV is found automatically: next to this file, in Milestone_2/, in
Data/telco_customer_churn_kaggle_dataset_raw/, or inside the zip in Data/.
"""

import argparse
import json
import sys
import tempfile
import time
import warnings
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Milestone_2"))

from eda_cleaning import load_and_clean  # noqa: E402
from feature_engineering import engineer_features  # noqa: E402

CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
SEED = 42
OUT_DIR = HERE / "outputs"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def find_data(tmp_dir: Path) -> Path:
    """Locate the raw CSV, extracting it from the Data/ zip if needed."""
    raw_dir = ROOT / "Data" / "telco_customer_churn_kaggle_dataset_raw"
    candidates = [
        HERE / CSV_NAME,
        ROOT / "Milestone_2" / CSV_NAME,
        raw_dir / CSV_NAME,
    ]
    for path in candidates:
        if path.exists():
            return path
    for zip_path in (ROOT / "Data").glob("*.zip"):
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if member.endswith(CSV_NAME):
                    zf.extract(member, tmp_dir)
                    return tmp_dir / member
    raise FileNotFoundError(f"Could not find {CSV_NAME}. Put it next to this script.")


def build_preprocessors(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()
    # Logistic Regression: same preprocessing as the Milestone 2 baseline.
    linear_prep = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ])
    # Tree models do not need scaling.
    tree_prep = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return linear_prep, tree_prep


# --------------------------------------------------------------------------
# Models and search spaces
# --------------------------------------------------------------------------
def make_pipeline(prep, estimator):
    # The "smote" step is switched on or off by the search, and sits inside the
    # pipeline so it is only ever fit on training folds (no leakage).
    return ImbPipeline([("prep", prep), ("smote", "passthrough"), ("model", estimator)])


def model_specs(linear_prep, tree_prep, pos_weight):
    smote_options = ["passthrough", SMOTE(random_state=SEED)]
    return {
        "Logistic Regression (tuned)": (
            make_pipeline(linear_prep, LogisticRegression(max_iter=2000, random_state=SEED)),
            {
                "model__C": loguniform(0.01, 10),
                "model__class_weight": [None, "balanced"],
                "smote": smote_options,
            },
        ),
        "Random Forest": (
            make_pipeline(tree_prep, RandomForestClassifier(random_state=SEED, n_jobs=-1)),
            {
                "model__n_estimators": randint(200, 600),
                "model__max_depth": [6, 8, 10, 12, 16, None],
                "model__min_samples_leaf": randint(1, 15),
                "model__min_samples_split": randint(2, 20),
                "model__max_features": ["sqrt", "log2", 0.3, 0.5],
                "model__class_weight": [None, "balanced", "balanced_subsample"],
                "smote": smote_options,
            },
        ),
        "XGBoost": (
            make_pipeline(tree_prep, XGBClassifier(
                eval_metric="logloss", tree_method="hist", random_state=SEED, n_jobs=-1)),
            {
                "model__n_estimators": randint(150, 600),
                "model__max_depth": randint(2, 7),
                "model__learning_rate": loguniform(0.01, 0.2),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.5, 0.5),
                "model__min_child_weight": randint(1, 10),
                "model__gamma": uniform(0, 2),
                "model__reg_lambda": loguniform(0.5, 10),
                "model__scale_pos_weight": [1.0, float(np.sqrt(pos_weight)), float(pos_weight)],
                "smote": smote_options,
            },
        ),
        "LightGBM": (
            make_pipeline(tree_prep, LGBMClassifier(
                subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1)),
            {
                "model__n_estimators": randint(150, 600),
                "model__num_leaves": randint(8, 64),
                "model__max_depth": [-1, 4, 6, 8],
                "model__learning_rate": loguniform(0.01, 0.2),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.5, 0.5),
                "model__min_child_samples": randint(10, 60),
                "model__reg_lambda": loguniform(0.5, 10),
                "model__class_weight": [None, "balanced"],
                "smote": smote_options,
            },
        ),
    }


# --------------------------------------------------------------------------
# Thresholds and evaluation
# --------------------------------------------------------------------------
def pick_thresholds(y_true, oof_proba, recall_target):
    """Choose thresholds from out-of-fold training predictions only."""
    precision, recall, thresholds = precision_recall_curve(y_true, oof_proba)
    p, r = precision[:-1], recall[:-1]
    f1 = 2 * p * r / (p + r + 1e-12)
    t_f1 = float(thresholds[int(np.argmax(f1))])
    ok = r >= recall_target
    t_recall = float(thresholds[ok].max()) if ok.any() else float(thresholds.min())
    return t_f1, t_recall


def metrics_at(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "Accuracy": (tp + tn) / len(y_true),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Missed churners": int(fn),
        "False alarms": int(fp),
    }


def evaluate(name, estimator, X_train, y_train, X_test, y_test, cv, recall_target, cv_score):
    """Return rows for three operating points plus the fitted model and test scores."""
    oof = cross_val_predict(estimator, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    t_f1, t_recall = pick_thresholds(y_train, oof, recall_target)

    estimator.fit(X_train, y_train)
    proba = estimator.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, proba)
    pr_auc = auc(recall, precision)  # same definition as the Milestone 2 baseline
    ap = average_precision_score(y_test, proba)
    roc = roc_auc_score(y_test, proba)

    rows = []
    for label, thr in [
        ("Default (0.50)", 0.50),
        ("Best-F1 threshold", t_f1),
        (f"Recall >= {recall_target:.2f} threshold", t_recall),
    ]:
        row = {"Model": name, "Operating point": label, "Threshold": round(thr, 3)}
        row.update(metrics_at(y_test, proba, thr))
        row.update({"PR-AUC": pr_auc, "Avg precision": ap, "ROC-AUC": roc, "CV avg precision": cv_score})
        rows.append(row)
    return rows, estimator, proba


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------
def md_table(df, float_cols):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.3f}" if c in float_cols and isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_plots(results, curves, importance, y_test):
    OUT_DIR.mkdir(exist_ok=True)

    # PR curves
    plt.figure(figsize=(7, 5))
    for name, proba in curves.items():
        p, r, _ = precision_recall_curve(y_test, proba)
        plt.plot(r, p, label=f"{name} ({auc(r, p):.3f})")
    plt.axhline(y_test.mean(), color="gray", linestyle=":", label=f"No skill ({y_test.mean():.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves (test set)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pr_curves.png", dpi=150)
    plt.close()

    # Model comparison at the best-F1 threshold
    best = results[results["Operating point"] == "Best-F1 threshold"].set_index("Model")
    metrics = ["F1", "Recall", "Precision", "PR-AUC"]
    ax = best[metrics].plot(kind="bar", figsize=(9, 5), width=0.8)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison at the best-F1 threshold (test set)")
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_comparison.png", dpi=150)
    plt.close()

    # Feature importance for the best tree model
    if importance is not None:
        name, imp = importance
        top = imp.sort_values(ascending=False).head(15)[::-1]
        plt.figure(figsize=(8, 6))
        plt.barh(top.index, top.values)
        plt.title(f"Top 15 features: {name}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "feature_importance.png", dpi=150)
        plt.close()


def write_summary(results, best_params, args, n_train, n_test, best_tree):
    show = ["Model", "Operating point", "Threshold", "F1", "Precision", "Recall",
            "Missed churners", "PR-AUC", "ROC-AUC"]
    floats = {"F1", "Precision", "Recall", "PR-AUC", "ROC-AUC"}
    lines = [
        "# Milestone 3: Advanced Model Tuning Results",
        "",
        f"Generated by `Milestone_3/model_tuning.py`. Train rows: {n_train}, test rows: {n_test}, "
        f"random_state {SEED}, {args.cv}-fold stratified CV, {args.n_iter} random search iterations per model.",
        "",
        "Features: cleaned data plus TenureGroup, NumServices and MonthlyChargePerService. "
        "Hyperparameters, the imbalance strategy and thresholds were chosen on the training set only; "
        "the test set was used once for these scores.",
        "",
        "## All results (test set)",
        "",
        md_table(results[show], floats),
        "",
        "PR-AUC is computed as in Milestone 2 (trapezoid area under the precision-recall curve). "
        "Thresholds come from out-of-fold training predictions.",
        "",
        f"Best tree model by cross-validated average precision: **{best_tree}**.",
        "",
        "## Best parameters",
        "",
        "```json",
        json.dumps(best_params, indent=2, default=str),
        "```",
        "",
        "Plots: `pr_curves.png`, `model_comparison.png`, `feature_importance.png`.",
    ]
    (OUT_DIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-iter", type=int, default=30, help="random search iterations per model")
    parser.add_argument("--cv", type=int, default=5, help="stratified CV folds")
    parser.add_argument("--recall-target", type=float, default=0.70)
    parser.add_argument("--quick", action="store_true", help="tiny search for a smoke test")
    parser.add_argument("--models", nargs="*", help="subset of models to run (substring match)")
    args = parser.parse_args()
    if args.quick:
        args.n_iter, args.cv = 4, 3

    OUT_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        data_path = find_data(Path(tmp))
        df = engineer_features(load_and_clean(data_path))

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"train {X_train.shape}, test {X_test.shape}, churn rate {y.mean():.1%}")

    linear_prep, tree_prep = build_preprocessors(X)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=SEED)

    all_rows, curves, best_params, cv_scores, fitted = [], {}, {}, {}, {}

    # Baseline: exactly the Milestone 2 configuration, rerun here so every
    # model is compared in the same environment on the same split.
    baseline = ImbPipeline([
        ("prep", linear_prep),
        ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    rows, _, proba = evaluate("Logistic Regression (M2 baseline)", baseline, X_train, y_train,
                              X_test, y_test, cv, args.recall_target, float("nan"))
    all_rows += rows
    curves["Logistic Regression (M2 baseline)"] = proba

    for name, (pipe, space) in model_specs(linear_prep, tree_prep, pos_weight).items():
        if args.models and not any(m.lower() in name.lower() for m in args.models):
            continue
        start = time.time()
        search = RandomizedSearchCV(
            pipe, space, n_iter=args.n_iter, scoring="average_precision",
            cv=cv, random_state=SEED, n_jobs=1, refit=True, error_score="raise")
        search.fit(X_train, y_train)
        params = {}
        for k, v in search.best_params_.items():
            if k == "smote":
                v = "SMOTE" if v != "passthrough" else "none"
            elif isinstance(v, np.generic):
                v = v.item()
            params[k] = v
        best_params[name] = params
        cv_scores[name] = float(search.best_score_)
        print(f"\n{name}: CV avg precision {search.best_score_:.4f} "
              f"({time.time() - start:.0f}s)\n  {params}")

        rows, est, proba = evaluate(name, search.best_estimator_, X_train, y_train, X_test,
                                    y_test, cv, args.recall_target, search.best_score_)
        all_rows += rows
        curves[name] = proba
        fitted[name] = est

    results = pd.DataFrame(all_rows)
    results.to_csv(OUT_DIR / "model_comparison.csv", index=False, float_format="%.4f")
    (OUT_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))

    # Feature importance for the best tree model (by CV average precision)
    tree_names = [n for n in fitted if n in ("Random Forest", "XGBoost", "LightGBM")]
    importance, best_tree = None, "n/a"
    if tree_names:
        best_tree = max(tree_names, key=lambda n: cv_scores[n])
        est = fitted[best_tree]
        names = [n.split("__", 1)[-1] for n in est.named_steps["prep"].get_feature_names_out()]
        importance = (best_tree, pd.Series(est.named_steps["model"].feature_importances_, index=names))

    save_plots(results, curves, importance, y_test)
    write_summary(results, best_params, args, len(X_train), len(X_test), best_tree)

    show = ["Model", "Operating point", "Threshold", "F1", "Precision", "Recall", "Missed churners", "PR-AUC"]
    print("\n" + results[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
