"""
train_innovation_model.py

Innovation model: Explainable Boosting Machine (EBM), via the
InterpretML library (`pip install interpret`).

Why this model, specifically, for this problem:
- The product's core promise is "show which parts of the address look
  suspicious" -- i.e. per-feature attribution is not a nice-to-have,
  it IS the product. An EBM is a Generalized Additive Model with
  automatically-detected pairwise interactions: prediction = sum of
  independent per-feature "shape functions" (+ a few interaction
  terms). That means feature attributions are EXACT and built into
  the model, not approximated after the fact (as SHAP does for the
  Random Forest).
- For a numeric feature like url_length, an EBM gives you the actual
  *shape* of risk vs. length (e.g. flat, then a sharp step up past
  ~75 characters), which a Logistic Regression coefficient cannot
  express (LR only gives a constant slope) and a Random Forest only
  exposes indirectly via feature_importances_ or partial dependence.
- EBMs are reported in the literature to reach accuracy close to
  gradient-boosted trees while remaining fully inspectable -- a
  genuinely different point on the accuracy/interpretability
  trade-off from both baselines, which is what this comparison tests.

This script trains the EBM on the SAME train/test split and the SAME
features as train_classification.py, and reports the same metrics
side by side with Logistic Regression and Random Forest so the
comparison is apples-to-apples.

Run as a module from the repo root:
    python -m phishing_detector.train_innovation_model
"""

import argparse
import os

import joblib
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def evaluate(name, y_test, y_pred, y_proba):
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
    return metrics


def main(features_path: str, model_dir: str):
    df = pd.read_csv(features_path)
    feature_cols = [c for c in df.columns if c not in ("url", "label")]
    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    results = []

    # Baseline 1: Logistic Regression (needs scaling)
    scaler = StandardScaler()
    X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train_s, y_train)
    results.append(evaluate("Logistic Regression", y_test, lr.predict(X_test_s),
                             lr.predict_proba(X_test_s)[:, 1]))

    # Baseline 2: Random Forest (no scaling needed)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=42, n_jobs=-1).fit(X_train, y_train)
    results.append(evaluate("Random Forest", y_test, rf.predict(X_test),
                             rf.predict_proba(X_test)[:, 1]))

    # Innovation model: Explainable Boosting Machine (no scaling needed)
    ebm = ExplainableBoostingClassifier(random_state=42, feature_names=feature_cols)
    ebm.fit(X_train, y_train)
    results.append(evaluate("Explainable Boosting Machine", y_test, ebm.predict(X_test),
                             ebm.predict_proba(X_test)[:, 1]))

    comparison = pd.DataFrame(results).set_index("model")
    print("\n=== Three-way comparison (identical train/test split) ===")
    print(comparison.to_string())

    # EBM's headline feature: exact global feature importance from the
    # additive structure of the model itself (not a post-hoc approximation).
    global_exp = ebm.explain_global()
    importances = pd.DataFrame({
        "feature": global_exp.data()["names"],
        "importance": global_exp.data()["scores"],
    }).sort_values("importance", ascending=False)
    print("\nEBM global feature importances (built-in, exact):")
    print(importances.head(10).to_string(index=False))

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(ebm, f"{model_dir}/ebm.joblib")
    comparison.to_csv(f"{model_dir}/three_way_comparison.csv")
    importances.to_csv(f"{model_dir}/ebm_feature_importance.csv", index=False)
    print(f"\nSaved EBM model and comparison tables to {model_dir}/")
    return comparison


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", default="data/processed/features.csv")
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()
    main(args.features_path, args.model_dir)


if __name__ == "__main__":
    cli()
