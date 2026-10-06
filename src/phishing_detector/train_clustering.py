"""Cluster class 1 training URLs separately for each dataset, fully offline."""

from __future__ import annotations

import argparse
import json
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from .split_data import FILES, file_hash, url_group
from .train_classification import FEATURES, POSITIVE_CLASS, load_partition


FLAGS = [name for name in FEATURES if name.startswith(("has_", "is_"))
         or name in ("uses_https", "brand_token_in_subdomain_or_path")]
COUNTS = [name for name in FEATURES if name.endswith(("_length", "_count"))]
OTHER = [name for name in FEATURES if name not in FLAGS + COUNTS]


def make_preprocessor() -> ColumnTransformer:
    """Reduce long count tails; keep rare binary flags from being amplified."""
    return ColumnTransformer([
        ("counts", Pipeline([
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]), COUNTS),
        ("other", StandardScaler(), OTHER),
        ("flags", "passthrough", FLAGS),
    ], remainder="drop")


def select_model(values: np.ndarray, seed: int, sample_size: int):
    """Compare k=2..6 on one fixed sample, fitting each model on all selected rows."""
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(len(values), min(sample_size, len(values)), replace=False))
    records, best_model, best_score = [], None, -np.inf
    for k in range(2, 7):
        model = MiniBatchKMeans(
            n_clusters=k, batch_size=4096, n_init=10,
            max_iter=100, max_no_improvement=20, random_state=seed,
        ).fit(values)
        labels = model.labels_
        counts = np.bincount(labels, minlength=k)
        sample_clusters = len(np.unique(labels[sample]))
        valid = counts.min() > 0 and 1 < sample_clusters < len(sample)
        score = float(silhouette_score(values[sample], labels[sample])) if valid else np.nan
        records.append({
            "k": k, "silhouette_estimate": score, "inertia": float(model.inertia_),
            "smallest_cluster_rows": int(counts.min()),
            "largest_cluster_share": float(counts.max() / len(labels)),
        })
        print(f"  k={k}: silhouette={score:.4f}; smallest cluster={counts.min():,}", flush=True)
        if np.isfinite(score) and score > best_score:
            best_model, best_score = model, score
    if best_model is None:
        raise ValueError("No usable clustering found; inspect feature diversity")
    return best_model, pd.DataFrame(records), sample


def profile_clusters(frame: pd.DataFrame, labels: np.ndarray):
    """Compare raw feature means against the entire selected class, in both directions."""
    overall = frame[FEATURES].mean()
    spread = frame[FEATURES].std(ddof=0).replace(0, 1)
    summaries, comparisons = [], []
    for cluster in sorted(np.unique(labels)):
        members = frame.loc[labels == cluster]
        mean = members[FEATURES].mean()
        difference = (mean - overall) / spread
        ranked = difference.abs().sort_values(ascending=False).index
        composition = members.label.value_counts()
        summaries.append({
            "cluster": int(cluster), "rows": len(members),
            "share_percent": 100 * len(members) / len(frame),
            "label_0_rows": int(composition.get(0, 0)),
            "label_1_rows": int(composition.get(1, 0)),
            "label_1_share": float(members.label.eq(1).mean()),
        })
        for rank, feature in enumerate(ranked, start=1):
            delta = float(difference[feature])
            comparisons.append({
                "cluster": int(cluster), "feature": feature, "rank": rank,
                "cluster_mean": float(mean[feature]), "overall_class_mean": float(overall[feature]),
                "standardized_difference": delta,
                "direction": "higher" if delta > 0 else "lower" if delta < 0 else "same",
            })
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)


def representative_examples(frame, labels, distances):
    """Choose up to three central examples from different domain groups per cluster."""
    examples = []
    for cluster in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == cluster)
        indices = indices[np.argsort(distances[indices], kind="stable")]
        seen = set()
        for index in indices:
            row = frame.iloc[index]
            domain = url_group(row.url)
            if domain in seen:
                continue
            seen.add(domain)
            examples.append({
                "cluster": int(cluster), "example_rank": len(seen),
                "training_row_number": int(row.training_row_number),
                "url": row.url, "domain_group": domain, "label": int(row.label),
                "distance_to_centroid": float(distances[index]),
                **{name: row[name] for name in FEATURES},
            })
            if len(seen) == 3:
                break
    return pd.DataFrame(examples)


def save_figures(destination, sizes, details, scores, chosen_k, title):
    """Export figures for inspection and the report; retain exact values in CSVs."""
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    bars = ax.bar(sizes.cluster.astype(str), sizes.rows, color="#1E40AF")
    ax.bar_label(bars, labels=[f"{value:,}" for value in sizes.rows], padding=3)
    ax.set(xlabel="Cluster", ylabel="Class 1 training URLs", title=f"{title}: cluster sizes")
    ax.margins(y=0.15)
    fig.savefig(destination / "cluster_sizes.png", dpi=180)
    plt.close(fig)

    table = details.pivot(index="feature", columns="cluster", values="standardized_difference")
    table = table.reindex(FEATURES)
    limit = max(1.0, float(np.abs(table.to_numpy()).max()))
    fig, ax = plt.subplots(figsize=(9, 10), layout="constrained")
    plot = ax.imshow(table.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(FEATURES)), [name.replace("_", " ") for name in FEATURES], fontsize=9)
    ax.set(xlabel="Cluster", title=f"{title}: feature differences from class average")
    fig.colorbar(plot, ax=ax, shrink=0.75, label="Mean difference / within-class standard deviation")
    fig.savefig(destination / "cluster_profiles.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(scores.k, scores.silhouette_estimate, "o-", color="#1E40AF")
    ax.axvline(chosen_k, color="#B45309", linestyle="--", label=f"Selected k={chosen_k}")
    ax.set(xlabel="Number of clusters", ylabel="Sample silhouette score", title=f"{title}: choosing k")
    ax.set_xticks(scores.k)
    ax.legend()
    fig.savefig(destination / "k_selection.png", dpi=180)
    plt.close(fig)


def cluster_dataset(dataset, split_root, out_root, seed, jobs, sample_size):
    """Read only training data and use labels solely for filtering/composition checks."""
    stem = Path(FILES[dataset]).stem
    source, destination = split_root / stem, out_root / stem
    if destination.exists():
        raise FileExistsError(f"Results already exist: {destination}. Use a new --out_dir.")
    split_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if split_summary["input_filename"] != FILES[dataset] or split_summary["sources_combined"]:
        raise ValueError("Split summary does not match this separate dataset")
    train = load_partition(source, "train", split_summary)
    train.insert(0, "training_row_number", np.arange(1, len(train) + 1))
    selected = train.loc[train.label.eq(1)].reset_index(drop=True)
    if len(selected) < 20 or (selected[COUNTS] < 0).any().any():
        raise ValueError("Need at least 20 class 1 rows with nonnegative counts")
    print(f"\n{FILES[dataset]}: clustering {len(selected):,} class 1 training URLs", flush=True)
    print(f"Class meaning: {POSITIVE_CLASS[dataset]}", flush=True)

    # Only these engineered URL columns enter either transformation or clustering.
    preprocessor = make_preprocessor()
    with threadpool_limits(limits=jobs):
        values = preprocessor.fit_transform(selected[FEATURES])
        model, scores, sample = select_model(values, seed, sample_size)
    labels = model.labels_
    distances = np.linalg.norm(values - model.cluster_centers_[labels], axis=1)
    sizes, details = profile_clusters(selected, labels)
    top = details.loc[details["rank"].le(3)].copy()
    examples = representative_examples(selected, labels, distances)
    assignments = selected[["training_row_number", "url", "label"]].copy()
    assignments["cluster"], assignments["distance_to_centroid"] = labels, distances
    pipeline = Pipeline([("preprocess", preprocessor), ("cluster", model)])

    # Write a complete result folder without overwriting earlier experiments.
    out_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="clustering_build_", dir=out_root) as temporary:
        staging = Path(temporary) / "ready"
        staging.mkdir()
        tables = {"cluster_summary": sizes, "cluster_feature_comparison": details,
                  "cluster_top_features": top, "cluster_examples": examples,
                  "cluster_assignments": assignments, "k_selection": scores,
                  "silhouette_sample": selected.iloc[sample][["training_row_number"]]}
        for name, table in tables.items():
            table.to_csv(staging / f"{name}.csv", index=False)
        joblib.dump(pipeline, staging / "clustering_pipeline.joblib", compress=3)
        title = "Malicious URLs" if dataset == "malicious_phish" else "PhiUSIIL"
        save_figures(staging, sizes, details, scores, model.n_clusters, title)
        summary = {
            "dataset": FILES[dataset], "class_label": 1, "class_meaning": POSITIVE_CLASS[dataset],
            "clustered_rows": len(selected), "selected_k": int(model.n_clusters), "seed": seed,
            "features": FEATURES, "labels_used_as_features": False, "held_out_sets_used": False,
            "sources_combined": False, "silhouette_sample_rows": len(sample),
            "selection_rule": "Highest silhouette estimate on the same fixed training sample",
            "preprocessing": {"log1p_then_standardize": COUNTS, "standardize": OTHER,
                              "binary_flags_unscaled": FLAGS},
            "model_parameters": model.get_params(),
            "input_train_sha256": split_summary["output_sha256"]["train.csv"],
            "split_summary_sha256": file_hash(source / "summary.json"),
            "code_sha256": file_hash(Path(__file__)),
            "package_versions": {name: metadata.version(name) for name in
                                 ("scikit-learn", "numpy", "pandas", "matplotlib", "joblib")},
            "interpretation_note": "Label composition is a filtering check, not proof of meaningful clusters",
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        staging.rename(destination)

    print(f"Selected k={model.n_clusters}", flush=True)
    print(sizes.to_string(index=False, float_format=lambda value: f"{value:.3f}"), flush=True)
    print(top[["cluster", "feature", "direction", "cluster_mean", "overall_class_mean"]]
          .to_string(index=False, float_format=lambda value: f"{value:.3f}"), flush=True)
    print(f"Saved profiles, examples, figures and model to {destination}\n", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["both", *FILES], default="both")
    parser.add_argument("--split_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--out_dir", type=Path, default=Path("clustering_output/grouped"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--silhouette_sample", type=int, default=2000)
    args = parser.parse_args()
    if args.jobs < 1 or args.silhouette_sample < 20:
        parser.error("Use positive --jobs and --silhouette_sample of at least 20")
    datasets = list(FILES) if args.dataset == "both" else [args.dataset]
    for dataset in datasets:
        stem = Path(FILES[dataset]).stem
        if not (args.split_dir / stem / "summary.json").is_file():
            parser.error(f"Missing splits for {dataset}; run split_data first")
        if (args.out_dir / stem).exists():
            parser.error(f"Results already exist: {args.out_dir / stem}. Use a new --out_dir.")
    for dataset in datasets:
        cluster_dataset(dataset, args.split_dir, args.out_dir,
                        args.seed, args.jobs, args.silhouette_sample)


if __name__ == "__main__":
    main()