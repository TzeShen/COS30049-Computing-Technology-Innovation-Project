"""
train_classification.py

Core detection task: binary classification of a URL as
phishing (1) vs legitimate (0), using only the offline, engineered
features from feature_extraction.py.

Models compared:
  - Logistic Regression : interpretable baseline (linear, per-feature
    coefficients map directly onto the app's "why is this flagged" UI)
  - Random Forest        : main classifier, captures feature interactions

Both are evaluated on the same held-out test split with the same
metrics so the comparison is fair.

Run as a module from the repo root:
    python -m phishing_detector.train_classification --threshold_sweep
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = None  # populated at runtime = all columns except url/label


def load_features(path: str):
    df = pd.read_csv(path)
    global FEATURE_COLS
    FEATURE_COLS = [c for c in df.columns if c not in ("url", "label")]
    X = df[FEATURE_COLS].values
    y = df["label"].values
    return df, X, y


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k:>10}: {v:.4f}")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["legit", "phishing"]))
    return metrics


def main(features_path: str, model_dir: str, threshold_sweep: bool):
    df, X, y = load_features(features_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Logistic Regression needs scaled inputs to converge reliably and for
    # coefficients to be comparable in magnitude across features.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=2000, class_weight="balanced")
    log_reg.fit(X_train_scaled, y_train)
    lr_metrics = evaluate("Logistic Regression", log_reg, X_test_scaled, y_test)

    # Logistic regression explainability: coefficients ranked by |weight|
    coef_table = pd.DataFrame({
        "feature": FEATURE_COLS,
        "coefficient": log_reg.coef_[0],
    }).sort_values("coefficient", key=np.abs, ascending=False)
    print("\nTop Logistic Regression coefficients (higher = pushes toward 'phishing'):")
    print(coef_table.head(10).to_string(index=False))

    # Random Forest works fine on raw (unscaled) features.
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_metrics = evaluate("Random Forest", rf, X_test, y_test)

    importances = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nTop Random Forest feature importances:")
    print(importances.head(10).to_string(index=False))

    if threshold_sweep:
        print("\n--- Random Forest: precision/recall at different thresholds ---")
        proba = rf.predict_proba(X_test)[:, 1]
        for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
            pred_t = (proba >= t).astype(int)
            print(f"threshold={t:.1f}  precision={precision_score(y_test, pred_t):.3f}  "
                  f"recall={recall_score(y_test, pred_t):.3f}  "
                  f"f1={f1_score(y_test, pred_t):.3f}")

    # Persist everything needed to run inference on a brand-new URL later.
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, f"{model_dir}/scaler.joblib")
    joblib.dump(log_reg, f"{model_dir}/logistic_regression.joblib")
    joblib.dump(rf, f"{model_dir}/random_forest.joblib")
    joblib.dump(FEATURE_COLS, f"{model_dir}/feature_columns.joblib")
    print(f"\nSaved scaler + both models to {model_dir}/")

    comparison = pd.DataFrame([lr_metrics, rf_metrics]).set_index("model")
    print("\n=== Side-by-side comparison ===")
    print(comparison.to_string())
    comparison.to_csv(f"{model_dir}/classification_comparison.csv")
    return comparison


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", default="data/processed/features.csv")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--threshold_sweep", action="store_true")
    args = parser.parse_args()
    main(args.features_path, args.model_dir, args.threshold_sweep)


if __name__ == "__main__":
    cli()
