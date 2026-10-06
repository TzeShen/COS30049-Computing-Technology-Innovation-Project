"""Create reproducible domain-grouped splits for each dataset separately."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
from functools import lru_cache
from importlib import metadata, resources
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit

import idna
import numpy as np
import pandas as pd
import tldextract

from .feature_extraction import extract_features


FILES = {
    "malicious_phish": "malicious_phish.csv",
    "phiusiil": "PhiUSIIL_Phishing_URL_Dataset.csv",
}
PARTS = ("train", "validation", "test")
SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")

# Use the bundled suffix list only. No HTTP requests or DNS lookups.
EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=500000)
def host_group(host: str) -> str:
    """Group equivalent host spellings and related subdomains together."""
    host = unquote(host).lower().rstrip(".")
    try:
        return "ip:" + ipaddress.ip_address(host).compressed
    except ValueError:
        pass
    try:
        host = idna.encode(host, uts46=True).decode("ascii")
    except idna.IDNAError:
        # Keep unusual host strings available instead of inventing a domain.
        pass
    result = EXTRACT(host)
    domain = result.top_domain_under_public_suffix
    # Unknown suffixes use the final two labels as a documented fallback.
    return "domain:" + (domain or ".".join(host.split(".")[-2:]))


def url_group(url: str) -> str:
    working = url if SCHEME.match(url) else "http://" + url
    host = urlsplit(working).hostname
    if not host:
        raise ValueError("A processed URL has no hostname; rerun data_prep")
    return host_group(host)


def choose_split(group: str, seed: int) -> str:
    """Stable assignment independent of row order, source and class labels."""
    digest = hashlib.sha256(f"{seed}|{group}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if bucket < 0.70 else "validation" if bucket < 0.85 else "test"


def split_dataset(input_path: Path, output_root: Path, seed: int) -> dict:
    destination = output_root / input_path.stem
    if destination.exists():
        raise FileExistsError(
            f"Splits already exist at {destination}. Keep them fixed for model "
            "comparisons, or choose a different --out_dir for a new experiment."
        )
    print(f"Reading and grouping {input_path.name}...", flush=True)
    frame = pd.read_csv(input_path, keep_default_na=False, float_precision="round_trip")
    feature_columns = list(extract_features("https://example.com").as_dict())
    if list(frame.columns) != ["url", "label", *feature_columns]:
        raise ValueError("Use the feature CSV from data/processed, with no metadata columns")
    if frame.empty or not frame.label.isin([0, 1]).all():
        raise ValueError("Expected nonempty processed data with labels 0 and 1")
    if not np.isfinite(frame[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("The feature table contains missing or non-finite values")

    groups = frame.url.map(url_group)
    allocation = {group: choose_split(group, seed) for group in groups.unique()}
    split = groups.map(allocation)
    group_sets = {name: set(groups[split == name]) for name in PARTS}
    for index, name in enumerate(PARTS):
        for other in PARTS[index + 1:]:
            if group_sets[name] & group_sets[other]:
                raise AssertionError("A domain group appears in multiple splits")

    stats = {}
    for name in PARTS:
        subset = frame.loc[split == name]
        labels = {str(key): int(value) for key, value in subset.label.value_counts().items()}
        if set(labels) != {"0", "1"}:
            raise ValueError(f"{name} does not contain both classes; review the grouping")
        stats[name] = {
            "rows": len(subset), "row_percent": round(100 * len(subset) / len(frame), 2),
            "domain_groups": len(group_sets[name]), "class_counts": labels,
        }
    if sum(value["rows"] for value in stats.values()) != len(frame):
        raise AssertionError("Split row counts do not match the input")

    # Store group/source-row references separately from model predictors.
    assignments = pd.DataFrame({
        "row_number": np.arange(1, len(frame) + 1),
        "domain_group": groups, "split": split,
    })
    assignments["split_row_number"] = assignments.groupby("split").cumcount() + 1
    snapshot = resources.files("tldextract").joinpath(".tld_set_snapshot").read_bytes()
    summary = {
        "input_filename": input_path.name, "input_sha256": file_hash(input_path),
        "seed": seed, "method": "SHA-256 of seed and normalized domain group",
        "target_group_proportions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "input_rows": len(frame), "domain_groups": int(groups.nunique()),
        "feature_columns": feature_columns, "splits": stats,
        "domain_overlap_between_splits": 0,
        "label_stratification": False, "sources_combined": False,
        "private_suffixes_included": True, "unknown_suffix_fallback": "final two hostname labels",
        "tldextract_version": metadata.version("tldextract"),
        "idna_version": metadata.version("idna"),
        "bundled_suffix_list_sha256": hashlib.sha256(snapshot).hexdigest(),
        "models_trained": False,
    }

    # Publish the folder only after all files are written successfully.
    output_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="split_build_", dir=output_root) as temporary:
        staging = Path(temporary) / "ready"
        staging.mkdir()
        for name in PARTS:
            frame.loc[split == name].to_csv(
                staging / f"{name}.csv", index=False, encoding="utf-8", lineterminator="\n"
            )
        assignments.to_csv(staging / "assignments.csv", index=False, lineterminator="\n")
        summary["output_sha256"] = {
            path.name: file_hash(path) for path in sorted(staging.glob("*.csv"))
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        staging.rename(destination)
    for name, value in stats.items():
        print(f"  {name}: {value['rows']:,} rows ({value['row_percent']}%)", flush=True)
    print(f"  Domain overlap: 0. Saved {destination}\n", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["both", *FILES], default="both")
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    names = list(FILES) if args.dataset == "both" else [args.dataset]
    paths = [args.processed_dir / FILES[name] for name in names]
    for path in paths:
        if not path.is_file():
            parser.error(f"Processed input not found: {path}")
        if (args.out_dir / path.stem).exists():
            parser.error(f"Splits already exist: {args.out_dir / path.stem}")
    for path in paths:
        split_dataset(path, args.out_dir, args.seed)


if __name__ == "__main__":
    main()