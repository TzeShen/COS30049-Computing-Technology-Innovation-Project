"""
train_clustering.py

Unsupervised clustering applied WITHIN a single class (phishing == 1).

Important constraints this script enforces:
  1. The label column is used ONLY to filter rows down to one class,
     and to report composition afterwards. It is never passed to the
     clustering algorithm and never used as a feature.
  2. Because every row already shares the same label, cluster quality
     cannot be judged by how well clusters predict the label (they all
     have the same one). Instead, each cluster is described by how its
     average feature values deviate from the overall (within-class)
     average -- this is where the actual structure is.

Models: K-Means (primary) and DBSCAN (comparison / outlier detection).

Run as a module from the repo root:
    python -m phishing_detector.train_clustering
"""

import argparse
import os

import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def choose_k(X_scaled, k_range=range(2, 8)):
    """Pick k by silhouette score; also print inertia (elbow) for reference."""
    print("\n--- Choosing k for K-Means ---")
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        scores[k] = sil
        print(f"k={k}: silhouette={sil:.4f}  inertia={km.inertia_:.1f}")
    best_k = max(scores, key=scores.get)
    print(f"Selected k={best_k} (highest silhouette score)")
    return best_k


def profile_clusters(df_class, feature_cols, cluster_labels, top_n=3):
    """
    For each cluster, compare mean feature values against the overall
    (within-class) average and report the top_n features with the
    largest absolute deviation, in either direction.
    """
    df = df_class.copy()
    df["cluster"] = cluster_labels
    overall_mean = df[feature_cols].mean()

    profiles = []
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        cluster_mean = sub[feature_cols].mean()
        # standardised deviation so features on different scales are comparable
        overall_std = df[feature_cols].std().replace(0, 1)
        deviation = (cluster_mean - overall_mean) / overall_std
        top_features = deviation.abs().sort_values(ascending=False).head(top_n)

        print(f"\n--- Cluster {c} (n={len(sub)}, "
              f"{len(sub) / len(df) * 100:.1f}% of clustered rows) ---")
        print("Label composition (sanity check, not used for clustering):")
        print(sub["label"].value_counts(normalize=True).round(3).to_dict())

        print(f"Top {top_n} features driving this cluster (z-score vs overall class average):")
        for feat in top_features.index:
            direction = "higher" if deviation[feat] > 0 else "lower"
            print(f"  {feat:<35} cluster mean={cluster_mean[feat]:.3f}  "
                  f"overall mean={overall_mean[feat]:.3f}  "
                  f"({direction}, z={deviation[feat]:+.2f})")

        print("Example URLs from this cluster:")
        for u in sub["url"].head(3):
            print(f"  - {u}")

        profiles.append({
            "cluster": c,
            "size": len(sub),
            "phishing_share": sub["label"].mean(),
            "top_features": list(top_features.index),
        })
    return pd.DataFrame(profiles)


def main(features_path: str, target_class: int, out_dir: str):
    df = pd.read_csv(features_path)
    feature_cols = [c for c in df.columns if c not in ("url", "label")]

    # --- Constraint: filter to a single class BEFORE clustering ---
    df_class = df[df["label"] == target_class].reset_index(drop=True)
    print(f"Clustering within class label={target_class}: {len(df_class)} rows")

    X = df_class[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- K-Means (primary clustering model) ---
    best_k = choose_k(X_scaled)
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    km_labels = kmeans.fit_predict(X_scaled)

    print("\n================ K-MEANS CLUSTER PROFILES ================")
    km_profile = profile_clusters(df_class, feature_cols, km_labels)

    # --- DBSCAN (comparison model: density-based, flags outliers as noise) ---
    # eps chosen via a quick heuristic on the scaled feature space;
    # in the report you'd normally justify this with a k-distance plot.
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    db_labels = dbscan.fit_predict(X_scaled)
    n_noise = int((db_labels == -1).sum())
    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    print(f"\n================ DBSCAN COMPARISON ================")
    print(f"DBSCAN found {n_clusters_db} clusters + {n_noise} noise points "
          f"({n_noise / len(df_class) * 100:.1f}% flagged as not fitting any dense group)")
    if n_clusters_db > 0:
        db_profile = profile_clusters(df_class, feature_cols, db_labels)
    else:
        db_profile = pd.DataFrame()

    os.makedirs(out_dir, exist_ok=True)
    df_class.assign(kmeans_cluster=km_labels, dbscan_cluster=db_labels).to_csv(
        f"{out_dir}/clustered_class_{target_class}.csv", index=False
    )
    km_profile.to_csv(f"{out_dir}/kmeans_cluster_profile.csv", index=False)
    if not db_profile.empty:
        db_profile.to_csv(f"{out_dir}/dbscan_cluster_profile.csv", index=False)
    print(f"\nSaved clustered rows and profiles to {out_dir}/")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", default="data/processed/features.csv")
    parser.add_argument("--target_class", type=int, default=1,
                         help="Which label to cluster within (default 1 = phishing)")
    parser.add_argument("--out_dir", default="clustering_output")
    args = parser.parse_args()
    main(args.features_path, args.target_class, args.out_dir)


if __name__ == "__main__":
    cli()
