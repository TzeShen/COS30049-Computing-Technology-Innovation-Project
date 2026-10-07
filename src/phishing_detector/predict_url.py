"""Offline inference with the validation-selected PhiUSIIL EBM.

This module only loads saved artifacts and predicts. It never fits a model,
changes its threshold, fetches a page, or looks up a domain.

Run from the project root:
    python -m phishing_detector.predict_url "https://example.org/account"
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd

from .feature_extraction import HOMOGLYPH_CHARS, extract_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "innovation" / "PhiUSIIL_Phishing_URL_Dataset"
MODEL_NAME = "PhiUSIIL EBM"
CLASSIFICATION_THRESHOLD = 0.5
EXPECTED_FEATURE_COUNT = 26
MAX_URL_LENGTH = 8192

# Display heuristics only. These cutoffs do not change the model or its score.
CONTINUOUS_THRESHOLDS = {
    "url_length": 60,
    "hostname_length": 30,
    "hyphen_count": 3,
    "hostname_hyphen_count": 2,
    "subdomain_count": 3,
    "entropy": 4.4,
    "digit_ratio": 0.2,
    "homoglyph_char_ratio": 0.35,
}

FEATURE_NAMES = {
    "url_length": "URL length",
    "hostname_length": "Hostname length",
    "path_length": "Path length",
    "query_length": "Query length",
    "digit_count": "Digit count",
    "digit_ratio": "Digit proportion",
    "special_char_count": "Special character count",
    "hyphen_count": "Hyphen count",
    "dot_count": "Dot count",
    "slash_count": "Slash count",
    "entropy": "Character entropy",
    "has_ip_address": "IP address host",
    "subdomain_count": "Subdomain count heuristic",
    "hostname_hyphen_count": "Hostname hyphens",
    "hostname_digit_count": "Hostname digits",
    "is_known_shortener": "Known URL shortener",
    "has_punycode": "Punycode hostname",
    "homoglyph_char_ratio": "Look-alike character proportion",
    "percent_encoded_count": "Percent sign count",
    "brand_token_in_subdomain_or_path": "Brand token outside the main domain",
    "tld_length": "Top-level domain length",
    "is_suspicious_tld": "TLD in the illustrative watchlist",
    "uses_https": "HTTPS scheme",
    "has_at_symbol": "At symbol",
    "has_port": "Explicit port",
    "has_double_slash_redirect": "Repeated slash pattern",
}


def load_artifacts(model_dir: str | Path = MODEL_DIR):
    """Load the selected EBM and reject mismatched features or class labels."""
    from interpret.glassbox import ExplainableBoostingClassifier

    folder = Path(model_dir)
    model = joblib.load(folder / "ebm.joblib")
    feature_cols = list(joblib.load(folder / "feature_columns.joblib"))
    if not isinstance(model, ExplainableBoostingClassifier):
        raise ValueError("The application requires the saved PhiUSIIL EBM.")
    if len(feature_cols) != EXPECTED_FEATURE_COUNT or len(set(feature_cols)) != len(feature_cols):
        raise ValueError("Expected the 26 distinct features used by the PhiUSIIL EBM.")
    if feature_cols != list(model.feature_names_in_):
        raise ValueError("Saved feature order does not match the EBM.")
    if set(feature_cols) != set(extract_features("https://example.org").as_dict()):
        raise ValueError("The feature extractor does not match the saved model's schema.")
    if list(model.classes_) != [0, 1] or model.link_ != "logit":
        raise ValueError("Expected class 0 = legitimate, class 1 = phishing, with a logit link.")
    return model, feature_cols


def artifact_signature(model_dir: str | Path = MODEL_DIR) -> str:
    """Identify the inference artifacts so legacy demo history stays separate."""
    digest = hashlib.sha256()
    for path in (
        Path(model_dir) / "ebm.joblib",
        Path(model_dir) / "feature_columns.joblib",
        Path(__file__).with_name("feature_extraction.py"),
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_url(raw_url: str) -> str:
    """Validate syntax locally, retaining the original URL representation."""
    url = raw_url.strip()
    if not url:
        raise ValueError("Enter a URL to analyse.")
    if len(url) > MAX_URL_LENGTH:
        raise ValueError(f"Please use a URL of at most {MAX_URL_LENGTH:,} characters.")
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in url):
        raise ValueError("The URL contains spaces or control characters. Use URL encoding where needed.")
    # Match the training extractor's handling of addresses without a scheme.
    # The added scheme is used for validation only, never for model features.
    working_url = url if re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", url) else "http://" + url
    try:
        parts = urlsplit(working_url)
        if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
            raise ValueError
        _ = parts.port
    except ValueError as error:
        raise ValueError("Enter a valid HTTP or HTTPS address with a hostname or IP address.") from error
    return url


def _feature_frame(feature_cols, feature_dict) -> pd.DataFrame:
    if set(feature_cols) != set(feature_dict):
        raise ValueError("Extracted features do not match the saved feature list.")
    frame = pd.DataFrame([feature_dict], columns=feature_cols)
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("URL features must be finite numeric values.")
    return frame


def _display_value(feature: str, value) -> str:
    if feature.startswith(("has_", "is_")) or feature in (
        "uses_https", "brand_token_in_subdomain_or_path"
    ):
        return "Yes" if value else "No"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.4f}"


def explain_prediction(model, feature_cols, feature_dict, top_n=6) -> list[dict]:
    """Return actual EBM contributions for this URL, including interactions.

    Contributions are log odds for class 1, not percentage points. Negative
    values and absent features are retained. Global importances are not used.
    See https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html
    """
    values = np.asarray(model.eval_terms(_feature_frame(feature_cols, feature_dict)))[0]
    if len(values) != len(model.term_names_) or not np.isfinite(values).all():
        raise ValueError("The EBM returned invalid local contributions.")
    terms = []
    for index, (name, value) in enumerate(zip(model.term_names_, values)):
        features = [feature_cols[i] for i in model.term_features_[index]]
        contribution = float(value)
        terms.append({
            "term": name,
            "title": " + ".join(FEATURE_NAMES.get(f, f) for f in features),
            "values": "; ".join(
                f"{FEATURE_NAMES.get(f, f)}: {_display_value(f, feature_dict[f])}"
                for f in features
            ),
            "interaction": len(features) > 1,
            "contribution": contribution,
            "direction": (
                "Raises the phishing score" if contribution > 0
                else "Lowers the phishing score" if contribution < 0
                else "No contribution"
            ),
        })
    terms.sort(key=lambda term: abs(term["contribution"]), reverse=True)
    return terms if top_n is None else terms[:top_n]


def build_indicators(features: dict) -> list[dict]:
    """Plain URL observations, kept separate from model contributions."""
    length = int(features["url_length"])
    subdomains = int(features["subdomain_count"])
    lookalike = float(features["homoglyph_char_ratio"])
    return [
        {"title": "IP address host", "flagged": bool(features["has_ip_address"]),
         "detail": "Detected" if features["has_ip_address"] else "Not detected"},
        {"title": "URL length", "flagged": length > CONTINUOUS_THRESHOLDS["url_length"],
         "detail": f"{length} characters; indicator cutoff is over 60"},
        {"title": "TLD watchlist", "flagged": bool(features["is_suspicious_tld"]),
         "detail": "In the illustrative watchlist" if features["is_suspicious_tld"] else "Not in the watchlist"},
        {"title": "Subdomain count heuristic",
         "flagged": subdomains >= CONTINUOUS_THRESHOLDS["subdomain_count"],
         "detail": f"{subdomains}; indicator cutoff is 3 or more"},
        {"title": "Punycode hostname", "flagged": bool(features["has_punycode"]),
         "detail": "Encoded hostname detected" if features["has_punycode"] else "Not detected"},
        {"title": "Look-alike character heuristic",
         "flagged": lookalike > CONTINUOUS_THRESHOLDS["homoglyph_char_ratio"],
         "detail": f"{lookalike:.1%} of hostname characters; cutoff is over 35%"},
    ]


def highlighted_url_parts(url: str, features: dict) -> list[dict]:
    """Return text segments and labels for Jinja to escape and highlight.

    No HTML is constructed from user input. Overlapping indicators can label
    the same segment, and concatenating all segments reproduces the URL.
    """
    spans = []
    scheme = re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", url)
    start = scheme.end() if scheme else 0
    end = min((i for i in range(start, len(url)) if url[i] in "/?#"), default=len(url))
    authority = url[start:end]
    host_start = start + authority.rfind("@") + 1
    host_port = url[host_start:end]
    host_end = (
        host_start + host_port.index("]") + 1
        if host_port.startswith("[") else host_start + len(host_port.split(":", 1)[0])
    )
    host = url[host_start:host_end]
    if features["has_ip_address"]:
        spans.append((host_start, host_end, "IP address host"))
    if features["is_suspicious_tld"] and "." in host:
        spans.append((host_start + host.rfind(".") + 1, host_end, "TLD watchlist"))
    if features["subdomain_count"] >= CONTINUOUS_THRESHOLDS["subdomain_count"]:
        labels = host.split(".")
        prefix_length = len(".".join(labels[:-2]))
        spans.append((host_start, host_start + prefix_length, "Subdomain count heuristic"))
    for match in re.finditer(r"[^.]+", host):
        if "xn--" in match.group().lower():
            spans.append((host_start + match.start(), host_start + match.end(), "Punycode hostname"))
    if features["homoglyph_char_ratio"] > CONTINUOUS_THRESHOLDS["homoglyph_char_ratio"]:
        for offset, char in enumerate(host):
            if char.lower() in HOMOGLYPH_CHARS:
                spans.append((host_start + offset, host_start + offset + 1, "Look-alike character heuristic"))
    cutoff = CONTINUOUS_THRESHOLDS["url_length"]
    if features["url_length"] > cutoff:
        spans.append((cutoff, len(url), "URL extends beyond 60 characters"))

    boundaries = sorted({0, len(url), *(point for left, right, _ in spans for point in (left, right))})
    parts = []
    for left, right in zip(boundaries, boundaries[1:]):
        labels = list(dict.fromkeys(label for a, b, label in spans if a <= left and right <= b))
        parts.append({"text": url[left:right], "highlighted": bool(labels), "label": "; ".join(labels)})
    return parts


def score_url(url: str, model_dir: str | Path = MODEL_DIR, *, model=None, feature_cols=None) -> dict:
    """Use the same saved model, feature order and threshold in CLI and Flask."""
    url = validate_url(url)
    if model is None and feature_cols is None:
        model, feature_cols = load_artifacts(model_dir)
    elif model is None or feature_cols is None:
        raise ValueError("Supply both the model and its feature columns.")
    features = extract_features(url).as_dict()
    frame = _feature_frame(feature_cols, features)
    phishing_score = float(model.predict_proba(frame)[0, list(model.classes_).index(1)])
    if not math.isfinite(phishing_score) or not 0 <= phishing_score <= 1:
        raise ValueError("The saved model returned an invalid score.")
    terms = explain_prediction(model, feature_cols, features, top_n=None)
    base = float(np.asarray(model.intercept_).reshape(-1)[0])
    total = base + math.fsum(term["contribution"] for term in terms)
    reconstructed = 1 / (1 + math.exp(-total)) if total >= 0 else math.exp(total) / (1 + math.exp(total))
    if not math.isclose(reconstructed, phishing_score, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError("The local explanation does not reconstruct the model score.")
    flagged = phishing_score >= CLASSIFICATION_THRESHOLD
    return {
        "url": url,
        "model_name": MODEL_NAME,
        "trust_score": round((1 - phishing_score) * 100, 1),
        # Retain this API key; the output is uncalibrated and the UI calls it a score.
        "phishing_probability": phishing_score,
        "risk_percent": round(phishing_score * 100, 1),
        "prediction": int(flagged),
        "status": "Potential phishing" if flagged else "Lower predicted risk",
        "status_style": "dangerous" if flagged else "safe",
        "threshold": CLASSIFICATION_THRESHOLD,
        "top_signals": [f"{term['title']}: {term['direction'].lower()}" for term in terms[:6]],
        "top_contributions": terms[:6],
        "model_contributions": terms,
        "base_log_odds": base,
        "total_log_odds": total,
        "raw_features": features,
        "indicators": build_indicators(features),
        "url_parts": highlighted_url_parts(url, features),
    }


def cli():
    if len(sys.argv) != 2:
        print('Usage: python -m phishing_detector.predict_url "<url>"')
        raise SystemExit(1)
    try:
        result = score_url(sys.argv[1])
    except (OSError, ImportError, ValueError, AttributeError) as error:
        print(f"Cannot analyse URL: {error}", file=sys.stderr)
        print(f"Expected saved EBM artifacts in: {MODEL_DIR}", file=sys.stderr)
        raise SystemExit(2) from error
    print(f"\nURL: {result['url']}")
    print(f"Model: {result['model_name']}")
    print(f"Trust score: {result['trust_score']:.1f} / 100")
    print(f"Model phishing score: {result['risk_percent']:.1f}%")
    print(f"Result: {result['status']} (fixed threshold: {result['threshold']})")
    print("The score is uncalibrated and does not establish whether a page is safe.")
    print("\nStrongest model contributions (log odds, not percentage points):")
    for term in result["top_contributions"]:
        print(f"  {term['contribution']:+.4f}  {term['title']}: {term['direction'].lower()}")
    print("\nDetected URL indicators (separate display heuristics):")
    active = [item for item in result["indicators"] if item["flagged"]]
    for indicator in active:
        print(f"  {indicator['title']}: {indicator['detail']}")
    if not active:
        print("  None of the displayed heuristics fired. The model can still flag the URL.")


if __name__ == "__main__":
    cli()
