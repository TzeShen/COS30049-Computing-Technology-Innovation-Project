"""Train separate baselines using fixed train/validation files. No test evaluation."""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .feature_extraction import extract_features
from .split_data import FILES, file_hash


FEATURES = list(extract_features("https://example.com").as_dict())
POSITIVE_CLASS = {
    "malicious_phish": "malicious: phishing, defacement or malware",
    "phiusiil": "phishing",
}


def load_partition(folder: Path, name: str, summary: dict) -> pd.DataFrame:
    """Verify the frozen split before selecting its engineered features."""
    path = folder / f"{name}.csv"
    if file_hash(path) != summary["output_sha256"][path.name]:
        raise ValueError(f"{path} changed after splitting. Restore the original split.")
    frame = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
    if list(frame.columns) != ["url", "label", *FEATURES]:
        raise ValueError(f"Unexpected columns in {path}")
    if set(frame.label.unique()) != {0, 1}:
        raise ValueError(f"Both classes, 0 and 1, are required in {path}")
    if len(frame) != summary["splits"][name]["rows"]:
        raise ValueError(f"Row count disagrees with the split summary: {path}")
    if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError(f"Missing or non-finite features in {path}")
    return frame


def make_models(seed: int, jobs: int) -> dict:
    """Use fixed starting settings; the dummy model is an imbalance reference."""
    return {
        "majority_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=2000,
                class_weight="balanced", random_state=seed,
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=jobs,
        ),
    }


def evaluate(y: np.ndarray, probability: np.ndarray) -> tuple[dict, np.ndarray]:
    """Evaluate label 1 at a fixed threshold; AP measures ranking quality."""
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y, probability),
        "average_precision": average_precision_score(y, probability),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return metrics, prediction


def train_dataset(dataset: str, split_root: Path, out_root: Path,
                  seed: int, jobs: int) -> pd.DataFrame:
    """Train one source independently and save validation evidence and models."""
    stem = Path(FILES[dataset]).stem
    folder, destination = split_root / stem, out_root / stem
    if destination.exists():
        raise FileExistsError(f"Results already exist: {destination}. Use a new --out_dir.")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary["input_filename"] != FILES[dataset] or summary["sources_combined"]:
        raise ValueError("The split summary does not match this separate dataset")
    if summary["domain_overlap_between_splits"] != 0:
        raise ValueError("Resolve domain overlap before training")

    # Test rows stay unread until the later, final model comparison.
    train = load_partition(folder, "train", summary)
    validation = load_partition(folder, "validation", summary)
    x_train, y_train = train[FEATURES].to_numpy(), train.label.to_numpy()
    x_val, y_val = validation[FEATURES].to_numpy(), validation.label.to_numpy()
    print(f"\n{FILES[dataset]}: {len(train):,} training / {len(validation):,} validation rows",
          flush=True)
    print(f"Positive class (1): {POSITIVE_CLASS[dataset]}", flush=True)

    models = make_models(seed, jobs)
    predictions = validation[["url", "label"]].copy()
    predictions.insert(0, "validation_row_number", np.arange(1, len(validation) + 1))
    results, settings = [], {}
    out_root.mkdir(parents=True, exist_ok=True)

    # Publish a complete result folder only after every model succeeds.
    with TemporaryDirectory(prefix="baseline_build_", dir=out_root) as temporary:
        staging = Path(temporary) / "ready"
        staging.mkdir()
        for name, model in models.items():
            print(f"  Training {name}...", flush=True)
            started = perf_counter()

            # Scaling is fitted only on training rows, inside the LR pipeline.
            with warnings.catch_warnings(), threadpool_limits(limits=jobs):
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(x_train, y_train)
            fit_seconds = perf_counter() - started
            positive_index = list(model.classes_).index(1)
            with threadpool_limits(limits=jobs):
                probability = model.predict_proba(x_val)[:, positive_index]
            scores, prediction = evaluate(y_val, probability)
            results.append({"model": name, "evaluation_split": "validation",
                            "threshold": 0.5, **scores, "fit_seconds": fit_seconds})
            predictions[f"{name}_probability"] = probability
            predictions[f"{name}_prediction"] = prediction
            joblib.dump(model, staging / f"{name}.joblib", compress=3)
            if isinstance(model, Pipeline):
                settings[name] = {key: step.get_params() for key, step in model.steps}
            else:
                settings[name] = model.get_params()
            print(f"    F1={scores['f1']:.4f}  recall={scores['recall']:.4f}  "
                  f"AP={scores['average_precision']:.4f}", flush=True)

        comparison = pd.DataFrame(results)
        comparison.to_csv(staging / "validation_metrics.csv", index=False)
        predictions.to_csv(staging / "validation_predictions.csv", index=False)
        joblib.dump(FEATURES, staging / "feature_columns.joblib")

        # LR coefficients refer to standardized inputs. RF importance is global,
        # not an explanation of why an individual URL received its prediction.
        pd.DataFrame({
            "feature": FEATURES,
            "coefficient": models["logistic_regression"].named_steps["classifier"].coef_[0],
        }).sort_values("coefficient", key=np.abs, ascending=False).to_csv(
            staging / "logistic_coefficients.csv", index=False
        )
        pd.DataFrame({
            "feature": FEATURES, "importance": models["random_forest"].feature_importances_,
        }).sort_values("importance", ascending=False).to_csv(
            staging / "random_forest_importances.csv", index=False
        )
        manifest = {
            "dataset": FILES[dataset], "positive_class": POSITIVE_CLASS[dataset],
            "seed": seed, "threshold": 0.5, "features": FEATURES,
            "training_rows": len(train), "validation_rows": len(validation),
            "test_evaluated": False, "hyperparameters_tuned": False,
            "sources_combined": False, "model_parameters": settings,
            "training_code_sha256": file_hash(Path(__file__)),
            "feature_extraction_sha256": file_hash(Path(__file__).with_name("feature_extraction.py")),
            "split_summary_sha256": file_hash(folder / "summary.json"),
            "input_sha256": {f"{part}.csv": summary["output_sha256"][f"{part}.csv"]
                             for part in ("train", "validation")},
            "python_version": platform.python_version(),
            "package_versions": {name: metadata.version(name) for name in
                                 ("scikit-learn", "numpy", "pandas", "joblib")},
        }
        (staging / "training_summary.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(destination)

    print(comparison[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]]
          .to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)
    print(f"Saved {destination}. Test set remains reserved.\n", flush=True)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["both", *FILES], default="both")
    parser.add_argument("--split_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--out_dir", type=Path, default=Path("models/baselines"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be a positive integer")
    datasets = list(FILES) if args.dataset == "both" else [args.dataset]
    for dataset in datasets:
        stem = Path(FILES[dataset]).stem
        if not (args.split_dir / stem / "summary.json").is_file():
            parser.error(f"Missing splits for {dataset}; run split_data first")
        if (args.out_dir / stem).exists():
            parser.error(f"Results already exist: {args.out_dir / stem}. Use a new --out_dir.")
    for dataset in datasets:
        train_dataset(dataset, args.split_dir, args.out_dir, args.seed, args.jobs)


if __name__ == "__main__":
    main()