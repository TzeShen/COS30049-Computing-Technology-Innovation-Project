"""Freeze validation choices, evaluate saved models, and export final test evidence."""

from __future__ import annotations

import argparse
import json
import math
import platform
import textwrap
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from threadpoolctl import threadpool_limits

from .split_data import FILES, file_hash
from .train_classification import FEATURES, POSITIVE_CLASS, evaluate, load_partition
from .train_website import subgroup_results


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare_dataset(dataset, args):
    """Read validation evidence and domain assignments before opening test rows."""
    stem = Path(FILES[dataset]).stem
    folder = args.split_dir / stem
    summary = read_json(folder / "summary.json")
    if summary["input_filename"] != FILES[dataset] or summary["sources_combined"]:
        raise ValueError("Expected separate dataset splits")
    if summary["feature_columns"] != FEATURES or summary["domain_overlap_between_splits"] != 0:
        raise ValueError("Unexpected split features or domain overlap")
    for filename in ("train.csv", "validation.csv", "assignments.csv"):
        if file_hash(folder / filename) != summary["output_sha256"][filename]:
            raise ValueError(f"Changed split input: {folder / filename}")
    assignments = pd.read_csv(folder / "assignments.csv", keep_default_na=False)
    if assignments.groupby("domain_group").split.nunique().gt(1).any():
        raise ValueError("A domain appears in multiple partitions")
    for part in ("train", "validation", "test"):
        rows = assignments.loc[assignments.split.eq(part)].sort_values("split_row_number")
        np.testing.assert_array_equal(rows.split_row_number, np.arange(1, summary["splits"][part]["rows"] + 1))

    groups = [
        (args.baseline_dir / stem, "training_summary.json",
         ["majority_baseline", "logistic_regression", "random_forest"], True),
        (args.innovation_dir / stem, "training_summary.json", ["ebm", "lightgbm"], False),
        (args.robustness_dir / stem, "summary.json", ["lightgbm_without_https"], False),
    ]
    models = []
    for root, manifest_name, names, numpy_input in groups:
        info = read_json(root / manifest_name)
        if info["dataset"] != FILES[dataset] or info["sources_combined"] or info["test_evaluated"]:
            raise ValueError(f"Invalid training provenance: {root}")
        if info["threshold"] != 0.5 or info["positive_class"] != POSITIVE_CLASS[dataset]:
            raise ValueError("Unexpected threshold or label meaning")
        for filename in ("train.csv", "validation.csv"):
            if info["input_sha256"][filename] != summary["output_sha256"][filename]:
                raise ValueError(f"Model used a different {filename}: {root}")
        expected = [x for x in FEATURES if x != "uses_https"] if len(names) == 1 else FEATURES
        if info["features"] != expected or joblib.load(root / "feature_columns.joblib") != expected:
            raise ValueError(f"Unexpected model feature order: {root}")
        for package in ("scikit-learn", "interpret-core", "lightgbm"):
            if package in info["package_versions"] and metadata.version(package) != info["package_versions"][package]:
                raise ValueError(f"Use the recorded {package} version from {root / manifest_name}")
        metrics = pd.read_csv(root / "validation_metrics.csv", float_precision="round_trip")
        for name in names:
            chosen = (metrics.loc[metrics.variant.eq("without_uses_https")]
                      if name == "lightgbm_without_https" else metrics.loc[metrics.model.eq(name)])
            if len(chosen) != 1 or chosen.iloc[0].evaluation_split != "validation" or chosen.iloc[0].threshold != 0.5:
                raise ValueError(f"Invalid validation row: {name}")
            row = chosen.iloc[0]
            models.append({
                "name": name, "training_dataset": dataset, "features": expected,
                "numpy_input": numpy_input, "path": str(root / f"{name}.joblib"),
                "model_sha256": file_hash(root / f"{name}.joblib"),
                "training_summary_sha256": file_hash(root / manifest_name),
                "validation_metrics_sha256": file_hash(root / "validation_metrics.csv"),
                "validation": {key: float(row[key]) for key in ("f1", "recall", "average_precision")},
            })
    selected = sorted(models, key=lambda x: (-x["validation"]["f1"], -x["validation"]["recall"],
                                             -x["validation"]["average_precision"], x["name"]))[0]["name"]
    return {"dataset": dataset, "folder": folder, "summary": summary, "assignments": assignments,
            "models": models, "selected": selected}


def get_test(bundle):
    frame = load_partition(bundle["folder"], "test", bundle["summary"])
    rows = bundle["assignments"].loc[bundle["assignments"].split.eq("test")].sort_values("split_row_number")
    frame.insert(0, "test_row_number", rows.split_row_number.to_numpy())
    frame.insert(1, "domain_group", rows.domain_group.to_numpy())
    return frame


def load_model(spec):
    path = Path(spec["path"])
    if file_hash(path) != spec["model_sha256"]:
        raise ValueError(f"Model changed after selection was frozen: {path}")
    model = joblib.load(path)
    if set(model.classes_) != {0, 1}:
        raise ValueError("Expected binary classes 0 and 1")
    if getattr(model, "n_features_in_", len(spec["features"])) != len(spec["features"]):
        raise ValueError("Model feature count disagrees with its manifest")
    return model


def model_input(frame, spec):
    x = frame[spec["features"]]
    return x.to_numpy() if spec["numpy_input"] else x


def save_errors(frame, probability, spec, folder, seed):
    """Sample up to three FP and three FN from distinct domains for manual review."""
    truth = frame.label.to_numpy()
    predicted = (probability >= 0.5).astype(int)
    rng = np.random.default_rng(seed)
    selected = []
    for mask in ((truth == 0) & (predicted == 1), (truth == 1) & (predicted == 0)):
        domains = set()
        for index in rng.permutation(np.flatnonzero(mask)):
            domain = frame.iloc[index].domain_group
            if domain not in domains:
                domains.add(domain)
                selected.append(int(index))
                if len(domains) == 3:
                    break
    selected.sort()
    examples = frame.iloc[selected].copy()
    examples["prediction"] = predicted[selected]
    examples["probability"] = probability[selected]
    examples["outcome"] = np.where(examples.label.eq(0), "FP", "FN")
    examples["manual_review_note"] = ""
    examples.to_csv(folder / f"{spec['name']}_error_examples.csv", index=False)
    if not selected:
        return
    model = load_model(spec)
    x = model_input(examples, spec)
    if hasattr(model, "eval_terms"):
        terms = model.eval_terms(x)
        names = model.term_names_
        indexes = model.term_features_
        base = np.full(len(examples), float(np.asarray(model.intercept_).reshape(-1)[0]))
        method = "EBM additive terms"
    elif hasattr(model, "booster_"):
        contributions = model.booster_.predict(x, pred_contrib=True, num_threads=2)
        terms, base = contributions[:, :-1], contributions[:, -1]
        names, indexes = spec["features"], [(i,) for i in range(len(spec["features"]))]
        method = "LightGBM TreeSHAP"
    else:
        return
    np.testing.assert_allclose(expit(base + terms.sum(axis=1)), probability[selected], rtol=1e-6, atol=1e-9)
    records = []
    for position, (_, row) in enumerate(examples.iterrows()):
        for rank, term in enumerate(np.argsort(-np.abs(terms[position]), kind="stable"), start=1):
            values = {spec["features"][i]: float(row[spec["features"][i]]) for i in indexes[term]}
            records.append({"test_row_number": int(row.test_row_number), "outcome": row.outcome,
                            "method": method, "term": names[term], "absolute_contribution_rank": rank,
                            "contribution_log_odds": float(terms[position, term]),
                            "base_log_odds": float(base[position]), "feature_values": json.dumps(values)})
    pd.DataFrame(records).to_csv(folder / f"{spec['name']}_error_contributions.csv", index=False)


def save_figures(truth, probabilities, metrics, folder):
    columns = min(3, len(probabilities))
    rows = math.ceil(len(probabilities) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.8 * rows), squeeze=False)
    for ax, (name, probability) in zip(axes.flat, probabilities.items()):
        ConfusionMatrixDisplay.from_predictions(truth, probability >= 0.5, labels=[0, 1],
                                                display_labels=["Class 0", "Class 1"],
                                                colorbar=False, ax=ax, values_format="d", cmap="Blues")
        ax.set_title(textwrap.fill(name.replace("_", " "), 25))
    for ax in list(axes.flat)[len(probabilities):]:
        ax.set_visible(False)
    fig.suptitle("Held-out test confusion matrices")
    fig.tight_layout()
    fig.savefig(folder / "confusion_matrices.png", dpi=180)
    plt.close(fig)
    for display, filename, title in (
        (PrecisionRecallDisplay, "precision_recall_curves.png", "Held-out test precision–recall curves"),
        (RocCurveDisplay, "roc_curves.png", "Held-out test ROC curves"),
    ):
        fig, ax = plt.subplots(figsize=(9, 6))
        for name, probability in probabilities.items():
            display.from_predictions(truth, probability, name=name.replace("_", " "), ax=ax)
            metric = "average_precision" if display is PrecisionRecallDisplay else "roc_auc"
            score = float(metrics.loc[metrics.model.eq(name), metric].iloc[0])
            label = "AP" if display is PrecisionRecallDisplay else "AUC"
            ax.lines[-1].set_label(f"{name.replace('_', ' ')} ({label} = {score:.4f})")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(folder / filename, dpi=180)
        plt.close(fig)


def evaluate_set(frame, specs, selected, folder, jobs, seed, context):
    folder.mkdir(parents=True)
    predictions = frame[["test_row_number", "domain_group", "url", "label", "uses_https"]].copy()
    records, probabilities, subgroups = [], {}, []
    truth = frame.label.to_numpy()
    if set(truth) != {0, 1}:
        raise ValueError("The declared evaluation needs both classes")
    for spec in specs:
        name = spec["name"]
        model = load_model(spec)
        with threadpool_limits(limits=jobs):
            probability = model.predict_proba(model_input(frame, spec))[:, list(model.classes_).index(1)]
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError(f"Invalid probabilities: {name}")
        scores, predicted = evaluate(truth, probability)
        records.append({"model": name, "training_dataset": spec["training_dataset"],
                        "evaluation_split": "test", "threshold": 0.5, "selected_on_validation": name in selected,
                        **scores, "validation_f1": spec["validation"]["f1"],
                        "f1_change_from_validation": scores["f1"] - spec["validation"]["f1"]})
        predictions[f"{name}_probability"] = probability
        predictions[f"{name}_prediction"] = predicted
        probabilities[name] = probability
        subgroups += subgroup_results(frame, probability, name)
        print(f"  {name}: F1={scores['f1']:.4f}  recall={scores['recall']:.4f}  "
              f"AP={scores['average_precision']:.4f}", flush=True)
        del model
    metrics = pd.DataFrame(records)
    metrics.to_csv(folder / "test_metrics.csv", index=False)
    predictions.to_csv(folder / "test_predictions.csv", index=False)
    pd.DataFrame(subgroups).to_csv(folder / "test_https_subgroups.csv", index=False)
    for spec in specs:
        if spec["name"] in selected:
            save_errors(frame, probabilities[spec["name"]], spec, folder, seed)
    save_figures(truth, probabilities, metrics, folder)
    write_json(folder / "summary.json", {**context, "test_rows": len(frame),
               "test_domain_groups": int(frame.domain_group.nunique()), "test_evaluated": True,
               "sources_combined": False, "models_refitted": False, "threshold": 0.5,
               "error_sampling": "Up to 3 FP and 3 FN per selected model, seeded, distinct domains",
               "manual_review_completed": False, "selected_for_error_review": selected})
    print(metrics[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]]
          .to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)
    return probabilities


def run(args):
    if args.out_dir.exists():
        raise FileExistsError(f"Results already exist: {args.out_dir}. Keep them; use a new --out_dir to reproduce.")
    bundles = {dataset: prepare_dataset(dataset, args) for dataset in FILES}
    primary, secondary = bundles["malicious_phish"], bundles["phiusiil"]
    for key in ("tldextract_version", "idna_version", "bundled_suffix_list_sha256",
                "private_suffixes_included", "unknown_suffix_fallback"):
        if primary["summary"][key] != secondary["summary"][key]:
            raise ValueError(f"Cross-source domain grouping differs: {key}")
    cross_models = []
    for alias, bundle in (("main_source_model", primary), ("phiusiil_model", secondary)):
        spec = next(x for x in bundle["models"] if x["name"] == bundle["selected"])
        cross_models.append({**spec, "original_model_name": spec["name"], "name": alias})
    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "threshold": 0.5,
        "selection_rule": "Highest validation F1; exact ties use recall, AP, then model name",
        "selection_completed_before_test_read": True,
        "datasets": {key: {"selected": value["selected"], "models": value["models"],
                           "split_summary_sha256": file_hash(value["folder"] / "summary.json"),
                           "test_sha256": value["summary"]["output_sha256"]["test.csv"]}
                     for key, value in bundles.items()},
        "cross_source": {"target": "phiusiil test", "models": cross_models,
                         "filter": "Exclude every domain group occurring anywhere in malicious_phish",
                         "interpretation": "Main model trained for broader malicious URLs; target has phishing and legitimate URLs",
                         "model_selection_uses_target_test": False},
        "evaluation_code_sha256": file_hash(Path(__file__)), "python_version": platform.python_version(),
        "package_versions": {name: metadata.version(name) for name in
                             ("scikit-learn", "interpret-core", "lightgbm", "numpy", "pandas", "joblib", "matplotlib")},
    }
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="final_evaluation_", dir=args.out_dir.parent) as temporary:
        staging = Path(temporary) / "ready"
        staging.mkdir()
        write_json(staging / "selection_before_test.json", plan)
        for dataset, bundle in bundles.items():
            print(f"\n{FILES[dataset]}: selected from validation = {bundle['selected']}", flush=True)
            frame = get_test(bundle)
            evaluate_set(frame, bundle["models"], [bundle["selected"]], staging / Path(FILES[dataset]).stem,
                         args.jobs, args.seed, {"dataset": FILES[dataset], "positive_class": POSITIVE_CLASS[dataset],
                                               "selection": bundle["selected"]})
            if dataset == "phiusiil":
                target_test = frame
        primary_domains = set(primary["assignments"].domain_group)
        transfer = target_test.loc[~target_test.domain_group.isin(primary_domains)].copy()
        if set(transfer.domain_group) & primary_domains:
            raise AssertionError("Source domain overlap in the transfer test")
        print(f"\nCross-source challenge: {len(transfer):,} PhiUSIIL test URLs; "
              f"excluded {len(target_test) - len(transfer):,} rows on shared domains", flush=True)
        evaluate_set(transfer, cross_models, [x["name"] for x in cross_models], staging / "cross_source",
                     args.jobs, args.seed, {"dataset": FILES["phiusiil"], "positive_class": POSITIVE_CLASS["phiusiil"],
                                           "excluded_shared_domain_rows": len(target_test) - len(transfer),
                                           "main_source_domain_overlap": 0, "models": cross_models,
                                           "interpretation": plan["cross_source"]["interpretation"]})
        write_json(staging / "completion.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
                   "test_evaluated": True, "models_refitted": False, "sources_combined": False,
                   "note": "Keep the validation selections fixed; report test failures without tuning on these test sets."})
        staging.rename(args.out_dir)
    print(f"\nSaved final evaluation to {args.out_dir}. Model selection remains based on validation.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--baseline_dir", type=Path, default=Path("models/baselines"))
    parser.add_argument("--innovation_dir", type=Path, default=Path("models/innovation"))
    parser.add_argument("--robustness_dir", type=Path, default=Path("models/robustness"))
    parser.add_argument("--out_dir", type=Path, default=Path("evaluation/final"))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling error examples only")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("Jobs must be a positive integer")
    run(args)


if __name__ == "__main__":
    main()