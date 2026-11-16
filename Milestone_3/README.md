# Milestone 3: Advanced Model Tuning

- `advanced_tuning.py`: first-pass Random Forest and XGBoost run (fixed hyperparameters).
- `model_tuning.py`: full tuning. Engineered features, cross-validated randomized search for Logistic Regression, Random Forest, XGBoost and LightGBM, imbalance handling (class weighting and SMOTE), and threshold tuning from out-of-fold predictions.
- `outputs/`: `results.md`, `model_comparison.csv`, `best_params.json`, `pr_curves.png`, `model_comparison.png`, `feature_importance.png`.

Run: `pip install -r Milestone_3/requirements.txt` then `python Milestone_3/model_tuning.py` (add `--quick` for a fast test). The full run is slow on a laptop because of the search.
