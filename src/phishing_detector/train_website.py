"""Check LightGBM's dependence on uses_https using frozen train/validation splits."""

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
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from .split_data import FILES, file_hash
from .train_classification import FEATURES, POSITIVE_CLASS, evaluate, load_partition


DROP = "uses_https"


def read_reference(folder, split_summary, validation):
    """Require the saved LightGBM experiment to use these exact partitions."""
    info = json.loads((folder / "training_summary.json").read_text(encoding="utf-8"))
    if info["dataset"] != split_summary["input_filename"] or info["features"] != FEATURES:
        raise ValueError("Reference dataset or features do not match")
    if info["sources_combined"] or info["test_evaluated"] or info["threshold"] != 0.5:
        raise ValueError("Expected separate, validation-only results at threshold 0.5")
    for part in ("train.csv", "validation.csv"):
        if info["input_sha256"][part] != split_summary["output_sha256"][part]:
            raise ValueError(f"Reference used a different {part}")
    version = info["package_versions"]["lightgbm"]
    if version != metadata.version("lightgbm"):
        raise ValueError(f"Use the reference LightGBM version: python -m pip install lightgbm=={version}")
    if info["training_protocol"]["lightgbm"] != (
        "Early stopping on saved validation binary log loss, patience 100"
    ):
        raise ValueError("Reference uses a different early-stopping protocol")
    saved = pd.read_csv(folder / "validation_predictions.csv",
                        keep_default_na=False, float_precision="round_trip")
    pd.testing.assert_frame_equal(validation[["url", "label"]], saved[["url", "label"]])
    np.testing.assert_array_equal(saved.validation_row_number, np.arange(1, len(saved) + 1))
    probability = saved.lightgbm_probability.to_numpy()
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid reference probabilities")
    scores, predicted = evaluate(validation.label.to_numpy(), probability)
    np.testing.assert_array_equal(predicted, saved.lightgbm_prediction)
    rows = pd.read_csv(folder / "validation_metrics.csv", float_precision="round_trip")
    rows = rows.loc[rows.model.eq("lightgbm")]
    if len(rows) != 1 or rows.iloc[0].evaluation_split != "validation" or rows.iloc[0].threshold != 0.5:
        raise ValueError("Expected one validation LightGBM result")
    for name, value in scores.items():
        if not np.isclose(value, rows.iloc[0][name]):
            raise ValueError(f"Reference {name} disagrees with its predictions")
    return info, probability, scores, float(rows.iloc[0].fit_seconds)


def subgroup_results(validation, probability, variant):
    """Keep undefined rates missing when a subgroup contains only one class."""
    truth = validation.label.to_numpy()
    predicted = (probability >= 0.5).astype(int)
    records = []
    for flag in (0, 1):
        mask = validation[DROP].to_numpy() == flag
        y, p = truth[mask], predicted[mask]
        tn = int(((y == 0) & (p == 0)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        tp = int(((y == 1) & (p == 1)).sum())
        records.append({
            "variant": variant, DROP: flag, "rows": len(y),
            "label_0_rows": tn + fp, "label_1_rows": tp + fn,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": (tn + tp) / len(y) if len(y) else np.nan,
            "recall": tp / (tp + fn) if tp + fn else np.nan,
            "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
        })
    return records


def train_dataset(dataset, split_root, reference_root, out_root, jobs):
    stem = Path(FILES[dataset]).stem
    source, reference, destination = split_root / stem, reference_root / stem, out_root / stem
    if destination.exists():
        raise FileExistsError(f"Results already exist: {destination}. Use a new --out_dir.")
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if summary["input_filename"] != FILES[dataset] or summary["sources_combined"]:
        raise ValueError("Split summary does not match this separate dataset")
    if summary["domain_overlap_between_splits"] != 0:
        raise ValueError("Resolve domain overlap before this experiment")
    train = load_partition(source, "train", summary)
    validation = load_partition(source, "validation", summary)
    if DROP not in FEATURES or any(not frame[DROP].isin([0, 1]).all() for frame in (train, validation)):
        raise ValueError("Expected the binary uses_https feature")
    info, original_probability, original_scores, original_seconds = read_reference(
        reference, summary, validation
    )
    features = [name for name in FEATURES if name != DROP]
    params = dict(info["model_parameters"]["lightgbm"])
    params["n_jobs"] = jobs
    model = LGBMClassifier(**params)
    weights = compute_sample_weight("balanced", train.label.to_numpy())

    print(f"\n{FILES[dataset]}: LightGBM with and without {DROP}", flush=True)
    print(f"Positive class (1): {POSITIVE_CLASS[dataset]}", flush=True)
    print(f"Training with {len(features)} features on {len(train):,} rows...", flush=True)
    started = perf_counter()
    with threadpool_limits(limits=jobs):
        model.fit(
            train[features], train.label.to_numpy(), sample_weight=weights,
            eval_X=validation[features], eval_y=validation.label.to_numpy(),
            eval_metric="binary_logloss",
            callbacks=[early_stopping(100, first_metric_only=True, verbose=False),
                       log_evaluation(period=0)],
        )
        fit_seconds = perf_counter() - started
        probability = model.predict_proba(validation[features])[:, list(model.classes_).index(1)]
    scores, prediction = evaluate(validation.label.to_numpy(), probability)
    variants = [
        ("all_features", FEATURES, original_scores, original_seconds),
        ("without_uses_https", features, scores, fit_seconds),
    ]
    metrics = pd.DataFrame([
        {"model": "lightgbm", "variant": name, "features": len(columns),
         "evaluation_split": "validation", "threshold": 0.5, **values,
         "f1_change": values["f1"] - original_scores["f1"], "fit_seconds": seconds}
        for name, columns, values, seconds in variants
    ])
    predictions = validation[["url", "label", DROP]].copy()
    predictions.insert(0, "validation_row_number", np.arange(1, len(validation) + 1))
    predictions["all_features_probability"] = original_probability
    predictions["all_features_prediction"] = (original_probability >= 0.5).astype(int)
    predictions["without_uses_https_probability"] = probability
    predictions["without_uses_https_prediction"] = prediction

    distribution = []
    for partition, frame in (("train", train), ("validation", validation)):
        for label in (0, 1):
            selected = frame.loc[frame.label.eq(label), DROP]
            distribution.append({"partition": partition, "label": label, "rows": len(selected),
                                 "https_rows": int(selected.sum()), "https_share": float(selected.mean())})
    subgroups = subgroup_results(validation, original_probability, "all_features")
    subgroups += subgroup_results(validation, probability, "without_uses_https")
    manifest = {
        "dataset": FILES[dataset], "positive_class": POSITIVE_CLASS[dataset],
        "model": "lightgbm", "features": features, "dropped_features": [DROP],
        "reference_features": FEATURES, "training_rows": len(train), "validation_rows": len(validation),
        "sources_combined": False, "test_evaluated": False, "threshold": 0.5,
        "model_parameters": model.get_params(deep=False), "best_iteration": int(model.best_iteration_),
        "training_protocol": info["training_protocol"]["lightgbm"],
        "weights": info["training_protocol"]["weights"],
        "comparison": "Same partitions, seed, class weighting, parameter budget and stopping rule",
        "limitations": ["One feature ablation; other features can retain source or scheme signals",
                        "Validation comparison, not a final test or proof of real-world safety"],
        "input_sha256": {part: summary["output_sha256"][part] for part in ("train.csv", "validation.csv")},
        "reference_sha256": {name: file_hash(reference / name) for name in
                             ("training_summary.json", "validation_metrics.csv", "validation_predictions.csv")},
        "training_code_sha256": file_hash(Path(__file__)),
        "feature_extraction_sha256": file_hash(Path(__file__).with_name("feature_extraction.py")),
        "python_version": platform.python_version(),
        "package_versions": {name: metadata.version(name) for name in
                             ("lightgbm", "scikit-learn", "numpy", "pandas", "joblib")},
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="robustness_build_", dir=out_root) as temporary:
        staging = Path(temporary) / "ready"
        staging.mkdir()
        joblib.dump(model, staging / "lightgbm_without_https.joblib", compress=3)
        joblib.dump(features, staging / "feature_columns.joblib")
        model.booster_.save_model(str(staging / "lightgbm_without_https.txt"))
        metrics.to_csv(staging / "validation_metrics.csv", index=False)
        predictions.to_csv(staging / "validation_predictions.csv", index=False)
        pd.DataFrame(distribution).to_csv(staging / "https_by_class.csv", index=False)
        pd.DataFrame(subgroups).to_csv(staging / "validation_subgroups.csv", index=False)
        pd.DataFrame({"feature": features, "gain": model.booster_.feature_importance(importance_type="gain")}) \
            .sort_values("gain", ascending=False).to_csv(staging / "feature_importance.csv", index=False)
        (staging / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(destination)
    print(metrics[["variant", "accuracy", "precision", "recall", "f1", "f1_change", "average_precision"]]
          .to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)
    print(f"Saved {destination}. Test data remains reserved.\n", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["both", *FILES], default="both")
    parser.add_argument("--split_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--reference_dir", type=Path, default=Path("models/innovation"))
    parser.add_argument("--out_dir", type=Path, default=Path("models/robustness"))
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("Jobs must be a positive integer")
    datasets = list(FILES) if args.dataset == "both" else [args.dataset]
    for dataset in datasets:
        stem = Path(FILES[dataset]).stem
        required = [args.split_dir / stem / "summary.json"] + [
            args.reference_dir / stem / name for name in
            ("training_summary.json", "validation_metrics.csv", "validation_predictions.csv")
        ]
        for path in required:
            if not path.is_file():
                parser.error(f"Missing input: {path}; complete innovation training first")
        if (args.out_dir / stem).exists():
            parser.error(f"Results already exist: {args.out_dir / stem}. Use a new --out_dir.")
    for dataset in datasets:
        train_dataset(dataset, args.split_dir, args.reference_dir, args.out_dir, args.jobs)


if __name__ == "__main__":
    main()
