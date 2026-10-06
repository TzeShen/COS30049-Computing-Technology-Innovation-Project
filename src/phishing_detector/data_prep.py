"""Prepare the two original URL datasets independently using offline signals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from .feature_extraction import extract_features


DATASETS = {
    "malicious_phish": {
        "filename": "malicious_phish.csv",
        "url_column": "url",
        "label_column": "type",
        "labels": {
            "benign": 0,
            "phishing": 1,
            "defacement": 1,
            "malware": 1,
        },
        "categories": {
            name: name
            for name in ("benign", "phishing", "defacement", "malware")
        },
    },
    "phiusiil": {
        "filename": "PhiUSIIL_Phishing_URL_Dataset.csv",
        "url_column": "URL",
        "label_column": "label",
        # Original PhiUSIIL labels use the reverse convention.
        "labels": {"1": 0, "0": 1},
        "categories": {"1": "benign", "0": "phishing"},
    },
}

SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def file_hash(path: Path) -> str:
    """Calculate a checksum for reproducibility."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_url(url: str) -> tuple[str, str, str]:
    """Validate locally and return a duplicate key, hostname and error reason."""
    if not url:
        return "", "", "empty_url"

    if CONTROL.search(url):
        return "", "", "literal_control_character"

    has_scheme = bool(SCHEME.match(url))

    try:
        parts = urlsplit(url if has_scheme else "http://" + url)
        host = parts.hostname or ""
        _ = parts.port
    except ValueError:
        return "", "", "url_or_port_parse_error"

    if not host:
        return "", "", "empty_hostname"

    if any(char.isspace() for char in host):
        return "", "", "whitespace_in_hostname"

    user, separator, host_port = parts.netloc.rpartition("@")
    netloc = (
        user + separator + host_port.lower()
        if separator
        else parts.netloc.lower()
    )

    # Missing schemes remain distinct from explicit HTTP or HTTPS.
    components = [
        parts.scheme.lower() if has_scheme else "missing",
        netloc,
        parts.path,
        parts.query,
        parts.fragment,
    ]

    key = hashlib.sha256(
        json.dumps(
            components,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    return key, host.lower().rstrip("."), ""


def counts(values: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in values.value_counts().items()
    }


def prepare_dataset(
    input_path: Path,
    output_path: Path,
    dataset: str,
) -> dict:
    """Write one feature table and its audit without combining sources."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.resolve() == output_path.resolve():
        raise ValueError(
            "Choose a processed output path different from the original input"
        )

    spec = DATASETS[dataset]
    url_column = spec["url_column"]
    label_column = spec["label_column"]

    # Read only raw URLs and labels, excluding supplied PhiUSIIL features.
    # The Python engine preserves NUL characters for explicit rejection.
    raw = pd.read_csv(
        input_path,
        usecols=[url_column, label_column],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        engine="python",
    )

    raw = raw.rename(
        columns={
            url_column: "url",
            label_column: "original_label",
        }
    )
    raw["source_row"] = range(1, len(raw) + 1)

    original_labels = raw.original_label.str.strip().str.lower()
    unknown = sorted(
        set(original_labels) - set(spec["labels"]) - {""}
    )

    if unknown:
        raise ValueError(
            f"Unexpected labels in {input_path.name}: {unknown}"
        )

    raw["label"] = original_labels.map(spec["labels"])
    raw["original_category"] = original_labels.map(spec["categories"])

    trimmed = int((raw.url != raw.url.str.strip()).sum())
    raw["url"] = raw.url.str.strip()

    details = pd.DataFrame(
        raw.url.map(inspect_url).tolist(),
        columns=["url_key", "hostname", "reason"],
    )
    raw = pd.concat([raw, details], axis=1)
    raw.loc[original_labels == "", "reason"] = "missing_label"

    invalid = raw.loc[raw.reason != ""].copy()
    valid = raw.loc[raw.reason == ""].copy()
    valid["label"] = valid.label.astype("int8")

    # Exclude every occurrence of URLs with conflicting binary labels.
    label_counts = valid.groupby("url_key", sort=False).label.nunique()
    conflict_keys = set(label_counts.index[label_counts > 1])

    conflicts = valid.loc[valid.url_key.isin(conflict_keys)].copy()
    conflicts["reason"] = "conflicting_binary_labels"

    eligible = valid.loc[~valid.url_key.isin(conflict_keys)].copy()
    selected = eligible.drop_duplicates("url_key", keep="first").copy()
    duplicates = eligible.loc[
        eligible.url_key.duplicated(keep="first")
    ].copy()

    # Keep successful URLs and their features aligned.
    feature_rows = []
    successful_indices = []
    failures = []
    feature_columns = tuple(
        extract_features("https://example.com").as_dict()
    )

    for index, url in selected.url.items():
        try:
            features = extract_features(url).as_dict()

            if tuple(features) != feature_columns or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in features.values()
            ):
                raise ValueError("Invalid feature schema or numeric value")

        except (ValueError, TypeError, OverflowError) as error:
            failures.append((index, type(error).__name__))
            continue

        successful_indices.append(index)
        feature_rows.append(features)

    if not successful_indices:
        raise ValueError(
            f"No usable URL records remain in {input_path.name}"
        )

    failed = selected.loc[
        [index for index, _ in failures]
    ].copy()
    failed["reason"] = [
        "feature_extraction_" + kind for _, kind in failures
    ]

    kept = selected.loc[successful_indices].reset_index(drop=True)
    kept["row_number"] = range(1, len(kept) + 1)

    features = pd.DataFrame(
        feature_rows,
        columns=feature_columns,
    )
    table = pd.concat(
        [kept[["url", "label"]], features],
        axis=1,
    )
    table["label"] = table.label.astype("int8")

    # Keep provenance outside predictors to prevent label/source leakage.
    metadata = kept[
        [
            "row_number",
            "source_row",
            "url_key",
            "hostname",
            "original_label",
            "original_category",
        ]
    ].copy()
    metadata.insert(1, "dataset", dataset)

    duplicate_members = eligible.loc[
        eligible.url_key.duplicated(keep=False)
    ]
    category_sets = (
        duplicate_members.groupby("url_key").original_category.agg(
            lambda values: ";".join(sorted(set(values)))
        )
    )

    metadata["original_category"] = (
        kept.url_key.map(category_sets).fillna(kept.original_category)
    )
    metadata["source_record_count"] = kept.url_key.map(
        eligible.url_key.value_counts()
    )
    metadata = metadata.rename(
        columns={"original_category": "original_categories"}
    )

    key_to_row = kept.set_index("url_key").row_number
    duplicates["retained_row_number"] = (
        duplicates.url_key.map(key_to_row).astype("Int64")
    )
    excluded = pd.concat(
        [invalid, conflicts, failed],
        ignore_index=True,
    )

    if len(raw) != len(excluded) + len(duplicates) + len(table):
        raise AssertionError("Cleaning counts do not reconcile")

    if not kept.url_key.is_unique or table.isna().any().any():
        raise AssertionError(
            "Feature output contains duplicates or missing values"
        )

    if len(metadata) != len(table):
        raise AssertionError("Metadata alignment failure")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir = output_path.parent / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    audit_columns = [
        "source_row",
        "url",
        "original_label",
        "original_category",
        "label",
    ]

    outputs = {
        output_path: table,
        audit_dir / f"{input_path.stem}_metadata.csv": metadata,
        audit_dir / f"{input_path.stem}_excluded.csv":
            excluded[audit_columns + ["reason"]],
        audit_dir / f"{input_path.stem}_duplicates.csv":
            duplicates[audit_columns + ["retained_row_number"]],
    }

    for path, frame in outputs.items():
        frame.to_csv(
            path,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )

    summary = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "input_filename": input_path.name,
        "output_filename": output_path.name,
        "input_sha256": file_hash(input_path),
        "input_rows": len(raw),
        "output_rows": len(table),
        "feature_count": len(feature_columns),
        "label_mapping": spec["labels"],
        "label_meanings": {"0": "benign", "1": "malicious"},
        "original_category_counts": counts(raw.original_category),
        "output_class_counts": counts(table.label),
        "outer_whitespace_trimmed_rows": trimmed,
        "invalid_rows": len(invalid),
        "conflicting_url_keys": len(conflict_keys),
        "conflicting_label_rows": len(conflicts),
        "duplicate_rows": len(duplicates),
        "feature_extraction_failures": len(failed),
        "exclusions_by_reason": counts(excluded.reason),
        "source_columns_used": [url_column, label_column],
        "feature_columns": list(feature_columns),
        "sources_combined": False,
        "cross_source_overlap_removed": False,
        "split_created": False,
        "models_trained": False,
        "verification": {
            "row_reconciliation": True,
            "unique_url_keys": True,
            "finite_numeric_features": True,
            "metadata_alignment": True,
        },
        "output_sha256": {
            str(path.relative_to(output_path.parent)): file_hash(path)
            for path in outputs
        },
    }

    summary_path = audit_dir / f"{input_path.stem}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"{input_path.name}: {len(raw):,} raw -> "
        f"{len(table):,} processed; "
        f"{len(feature_columns)} features. Saved {output_path}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["both", *DATASETS],
        default="both",
        help="Process one source or both independently",
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/raw"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--in_path",
        type=Path,
        help="Optional input override for one dataset",
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        help="Optional output override for one dataset",
    )
    args = parser.parse_args()

    if args.dataset == "both" and (args.in_path or args.out_path):
        parser.error(
            "Select --dataset malicious_phish or phiusiil "
            "when overriding file paths"
        )

    choices = (
        list(DATASETS)
        if args.dataset == "both"
        else [args.dataset]
    )

    jobs = [
        (
            name,
            args.in_path or args.raw_dir / DATASETS[name]["filename"],
            args.out_path or args.out_dir / DATASETS[name]["filename"],
        )
        for name in choices
    ]

    for _, input_path, output_path in jobs:
        if not input_path.is_file():
            parser.error(f"Input not found: {input_path}")

        if input_path.resolve() == output_path.resolve():
            parser.error("Raw and processed paths must differ")

    for dataset, input_path, output_path in jobs:
        print(f"Processing {input_path.name}...", flush=True)
        prepare_dataset(input_path, output_path, dataset)


if __name__ == "__main__":
    main()