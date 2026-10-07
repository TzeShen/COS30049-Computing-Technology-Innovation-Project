"""Train EBM and LightGBM on each frozen dataset split. Test data stays reserved."""

from __future__ import annotations

import argparse
import json
import platform
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.special import expit
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from .split_data import FILES, file_hash
from .train_classification import FEATURES, POSITIVE_CLASS, evaluate, load_partition


def load_baselines(folder, split_summary):
    """Compare only results produced from exactly the same input partitions."""
    info = json.loads((folder / "training_summary.json").read_text(encoding="utf-8"))
    if info["dataset"] != split_summary["input_filename"] or info["features"] != FEATURES:
        raise ValueError("Baseline dataset/features do not match this experiment")
    if info["sources_combined"] or info["test_evaluated"]:
        raise ValueError("Use the separate, validation-only baseline results")
    for part in ("train.csv", "validation.csv"):
        if info["input_sha256"][part] != split_summary["output_sha256"][part]:
            raise ValueError(f"Baseline used a different {part}; do not compare different splits")
    metrics = pd.read_csv(folder / "validation_metrics.csv", float_precision="round_trip")
    expected = {"majority_baseline", "logistic_regression", "random_forest"}
    if len(metrics) != 3 or set(metrics.model) != expected:
        raise ValueError("Expected the three previously saved baseline rows")
    if not metrics.evaluation_split.eq("validation").all() or not metrics.threshold.eq(0.5).all():
        raise ValueError("Baseline metrics must use validation data and threshold 0.5")
    return metrics


def make_models(seed, jobs, ebm_rounds, lightgbm_rounds):
    """Use a compact additive model and a more flexible boosted-tree comparison."""
    return {
        "ebm": ExplainableBoostingClassifier(
            feature_names=FEATURES, feature_types=["continuous"] * len(FEATURES),
            interactions=10, max_bins=128, max_interaction_bins=16,
            learning_rate=0.03, max_rounds=ebm_rounds, min_samples_leaf=20,
            reg_lambda=1.0, outer_bags=1, validation_size=0,
            early_stopping_rounds=0, random_state=seed, n_jobs=1,
        ),
        "lightgbm": LGBMClassifier(
            objective="binary", n_estimators=lightgbm_rounds, learning_rate=0.05,
            num_leaves=31, min_child_samples=50, reg_lambda=1.0,
            subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
            random_state=seed, n_jobs=jobs, deterministic=True,
            force_col_wise=True, verbosity=-1,
        ),
    }


def save_ebm_examples(model, validation, probability, destination, seed):
    """Export actual local contributions for up to three examples per TP/TN/FP/FN."""
    truth = validation.label.to_numpy()
    predicted = (probability >= 0.5).astype(int)
    outcomes = np.select(
        [(truth == 0) & (predicted == 0), (truth == 0) & (predicted == 1),
         (truth == 1) & (predicted == 0)], ["TN", "FP", "FN"], default="TP"
    )
    rng = np.random.default_rng(seed)
    selected = []
    for outcome in ("TN", "FP", "FN", "TP"):
        candidates = np.flatnonzero(outcomes == outcome)
        selected.extend(rng.choice(candidates, min(3, len(candidates)), replace=False).tolist())
    selected = sorted(selected)
    terms = model.eval_terms(validation.iloc[selected][FEATURES])
    intercept = float(np.asarray(model.intercept_).reshape(-1)[0])
    if not np.allclose(expit(intercept + terms.sum(axis=1)), probability[selected]):
        raise AssertionError("EBM contributions do not reconstruct the prediction")
    records = []
    for position, row_index in enumerate(selected):
        row = validation.iloc[row_index]
        order = np.argsort(-np.abs(terms[position]), kind="stable")
        for rank, term_index in enumerate(order, start=1):
            feature_values = {FEATURES[index]: float(row[FEATURES[index]])
                              for index in model.term_features_[term_index]}
            records.append({
                "validation_row_number": row_index + 1, "url": row.url,
                "label": int(row.label), "prediction": int(predicted[row_index]),
                "outcome": outcomes[row_index], "probability": float(probability[row_index]),
                "term": model.term_names_[term_index], "absolute_contribution_rank": rank,
                "contribution_log_odds": float(terms[position, term_index]),
                "intercept_log_odds": intercept, "feature_values": json.dumps(feature_values),
            })
    pd.DataFrame(records).to_csv(destination / "ebm_local_examples.csv", index=False)


def train_dataset(dataset, split_root, baseline_root, out_root,
                  seed, jobs, ebm_rounds, lightgbm_rounds):
    """Fit each source independently and compare its validation results with its baselines."""
    stem = Path(FILES[dataset]).stem
    source, baseline, destination = split_root / stem, baseline_root / stem, out_root / stem
    if destination.exists():
        raise FileExistsError(f"Results already exist: {destination}. Use a new --out_dir.")
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if summary["input_filename"] != FILES[dataset] or summary["sources_combined"]:
        raise ValueError("Split summary does not match this separate dataset")
    if summary["domain_overlap_between_splits"] != 0:
        raise ValueError("Resolve domain overlap before training")
    baseline_metrics = load_baselines(baseline, summary)

    # The verified loader checks file hashes, feature columns, labels and finite values.
    train = load_partition(source, "train", summary)
    validation = load_partition(source, "validation", summary)
    x_train, y_train = train[FEATURES], train.label.to_numpy()
    x_val, y_val = validation[FEATURES], validation.label.to_numpy()
    weights = compute_sample_weight("balanced", y_train)
    print(f"\n{FILES[dataset]}: {len(train):,} training / {len(validation):,} validation rows",
          flush=True)
    print(f"Positive class (1): {POSITIVE_CLASS[dataset]}", flush=True)

    models = make_models(seed, jobs, ebm_rounds, lightgbm_rounds)
    predictions = validation[["url", "label"]].copy()
    predictions.insert(0, "validation_row_number", np.arange(1, len(validation) + 1))
    results = []
    out_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="innovation_build_", dir=out_root) as temporary:
        staging = Path(temporary) / "ready"
        staging.mkdir()
        for name, model in models.items():
            print(f"  Training {name}...", flush=True)
            started = perf_counter()
            with threadpool_limits(limits=jobs):
                if name == "ebm":
                    # Fixed rounds avoid creating a hidden random validation split.
                    model.fit(x_train, y_train, sample_weight=weights)
                else:
                    # Validation controls early stopping; test data is never loaded.
                    model.fit(
                        x_train, y_train, sample_weight=weights,
                        eval_X=x_val, eval_y=y_val, eval_metric="binary_logloss",
                        callbacks=[early_stopping(100, first_metric_only=True, verbose=False),
                                   log_evaluation(period=0)],
                    )
                fit_seconds = perf_counter() - started
                positive_index = list(model.classes_).index(1)
                probability = model.predict_proba(x_val)[:, positive_index]
            scores, prediction = evaluate(y_val, probability)
            results.append({"model": name, "evaluation_split": "validation", "threshold": 0.5,
                            **scores, "fit_seconds": fit_seconds})
            predictions[f"{name}_probability"] = probability
            predictions[f"{name}_prediction"] = prediction
            joblib.dump(model, staging / f"{name}.joblib", compress=3)
            print(f"    F1={scores['f1']:.4f}  recall={scores['recall']:.4f}  "
                  f"AP={scores['average_precision']:.4f}  fit={fit_seconds:.1f}s", flush=True)
            if name == "ebm":
                importance = pd.DataFrame({"term": model.term_names_,
                                           "importance": model.term_importances()})
                importance.sort_values("importance", ascending=False).to_csv(
                    staging / "ebm_term_importance.csv", index=False
                )
                save_ebm_examples(model, validation, probability, staging, seed)
            else:
                # Global split gain is not an explanation of an individual prediction.
                pd.DataFrame({"feature": FEATURES,
                              "gain": model.booster_.feature_importance(importance_type="gain")}) \
                    .sort_values("gain", ascending=False).to_csv(
                        staging / "lightgbm_feature_importance.csv", index=False
                    )
                model.booster_.save_model(str(staging / "lightgbm.txt"))

        metrics = pd.DataFrame(results)
        comparison = pd.concat([baseline_metrics, metrics], ignore_index=True)
        metrics.to_csv(staging / "validation_metrics.csv", index=False)
        comparison.to_csv(staging / "validation_comparison.csv", index=False)
        predictions.to_csv(staging / "validation_predictions.csv", index=False)
        joblib.dump(FEATURES, staging / "feature_columns.joblib")
        manifest = {
            "dataset": FILES[dataset], "positive_class": POSITIVE_CLASS[dataset],
            "seed": seed, "features": FEATURES, "threshold": 0.5,
            "training_rows": len(train), "validation_rows": len(validation),
            "sources_combined": False, "test_evaluated": False,
            "model_parameters": {name: model.get_params(deep=False) for name, model in models.items()},
            "training_protocol": {
                "ebm": "Fixed rounds, one bag, no internal validation or early stopping",
                "lightgbm": "Early stopping on saved validation binary log loss, patience 100",
                "weights": "Balanced class weights computed from training labels only",
                "threshold_tuned": False, "probabilities_calibrated": False,
            },
            "ebm_best_iteration_by_stage_and_bag": np.asarray(models["ebm"].best_iteration_).tolist(),
            "lightgbm_best_iteration": int(models["lightgbm"].best_iteration_),
            "input_sha256": {part: summary["output_sha256"][part]
                             for part in ("train.csv", "validation.csv")},
            "split_summary_sha256": file_hash(source / "summary.json"),
            "baseline_metrics_sha256": file_hash(baseline / "validation_metrics.csv"),
            "training_code_sha256": file_hash(Path(__file__)),
            "feature_extraction_sha256": file_hash(Path(__file__).with_name("feature_extraction.py")),
            "python_version": platform.python_version(),
            "package_versions": {name: metadata.version(name) for name in
                                 ("interpret-core", "lightgbm", "scikit-learn", "numpy", "pandas", "joblib")},
            "local_explanation_units": "Log odds for class 1; positive values increase the model score",
        }
        (staging / "training_summary.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(destination)
    print(comparison[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]]
          .to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)
    print(f"Saved {destination}. These are validation results; test data remains reserved.\n", flush=True)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["both", *FILES], default="both")
    parser.add_argument("--split_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--baseline_dir", type=Path, default=Path("models/baselines"))
    parser.add_argument("--out_dir", type=Path, default=Path("models/innovation"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--ebm_rounds", type=int, default=1000)
    parser.add_argument("--lightgbm_rounds", type=int, default=1500)
    args = parser.parse_args()
    if min(args.jobs, args.ebm_rounds, args.lightgbm_rounds) < 1:
        parser.error("Jobs and round limits must be positive integers")
    datasets = list(FILES) if args.dataset == "both" else [args.dataset]
    for dataset in datasets:
        stem = Path(FILES[dataset]).stem
        required = [args.split_dir / stem / "summary.json",
                    args.baseline_dir / stem / "training_summary.json",
                    args.baseline_dir / stem / "validation_metrics.csv"]
        for path in required:
            if not path.is_file():
                parser.error(f"Missing input: {path}; complete splitting and baseline training first")
        if (args.out_dir / stem).exists():
            parser.error(f"Results already exist: {args.out_dir / stem}. Use a new --out_dir.")
    for dataset in datasets:
        train_dataset(dataset, args.split_dir, args.baseline_dir, args.out_dir,
                      args.seed, args.jobs, args.ebm_rounds, args.lightgbm_rounds)


if __name__ == "__main__":
    main()
