"""
data_prep.py

Loads a raw (url, label) dataset, applies feature_extraction to every
URL, cleans the result, and writes a model-ready feature table.

Cleaning steps:
- Drop exact duplicate URLs (common when combining multiple raw sources).
- Drop rows where the URL failed to parse into any host at all.
- Report and drop rows with missing/blank labels.
- No missing numeric values are possible by construction (every feature
  defaults to 0 on parse failure), but we assert this explicitly so a
  silent NaN can never slip into training.

Run as a module from the repo root:
    python -m phishing_detector.data_prep
"""

import argparse
import pandas as pd

from .feature_extraction import extract_features


def build_feature_table(raw_csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv_path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    assert {"url", "label"}.issubset(raw.columns), \
        "raw csv must have 'url' and 'label' columns"

    before = len(raw)
    raw = raw.dropna(subset=["url", "label"])
    raw["url"] = raw["url"].astype(str).str.strip()
    raw = raw[raw["url"] != ""]
    raw = raw.drop_duplicates(subset=["url"])
    print(f"Rows: {before} raw -> {len(raw)} after dropping missing/blank/duplicate URLs")

    feature_rows = []
    bad_rows = 0
    for url in raw["url"]:
        try:
            feature_rows.append(extract_features(url).as_dict())
        except Exception:
            bad_rows += 1
            feature_rows.append(None)

    features = pd.DataFrame(feature_rows)
    mask = features.notna().all(axis=1)
    if bad_rows:
        print(f"Dropped {bad_rows} URLs that failed to parse")

    out = pd.concat(
        [raw.reset_index(drop=True)[mask.reset_index(drop=True)],
         features[mask].reset_index(drop=True)],
        axis=1,
    )
    out["label"] = out["label"].astype(int)

    assert out.isna().sum().sum() == 0, "unexpected missing values after feature extraction"
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", default="data/raw/raw_urls.csv")
    parser.add_argument("--out_path", default="data/processed/features.csv")
    args = parser.parse_args()

    table = build_feature_table(args.in_path)
    table.to_csv(args.out_path, index=False)
    print(f"Wrote {len(table)} rows x {table.shape[1]} columns to {args.out_path}")
    print(table["label"].value_counts().rename({0: "legitimate", 1: "phishing"}))


if __name__ == "__main__":
    main()
