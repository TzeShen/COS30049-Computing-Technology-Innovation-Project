"""
predict_url.py

This is the piece that actually backs the application described in the
brief: paste a URL in, get a trust score back plus a human-readable
explanation of which signals drove it. Fully offline -- loads saved
models from disk and does local feature extraction only; no requests
are made to the pasted URL or anywhere else.

Usage (from the repo root, after training has produced models/):
    python -m phishing_detector.predict_url "http://secure-paypal-login.xyz/account"
"""

import sys

import joblib
import numpy as np

from .feature_extraction import extract_features

MODEL_DIR = "models"

# Thresholds below which a continuous feature is considered "normal" and
# therefore not worth surfacing as a suspicious signal. These are rough,
# illustrative cutoffs (roughly typical values for legitimate URLs) --
# in a full project you'd derive these from the training data's
# legitimate-class distribution (e.g. the 90th percentile) rather than
# hardcoding them.
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

FRIENDLY_NAMES = {
    "has_ip_address": "uses a raw IP address instead of a domain name",
    "is_suspicious_tld": "uses a top-level domain often associated with abuse (e.g. .xyz, .top, .click)",
    "has_punycode": "contains punycode (xn--), often used to fake look-alike characters",
    "homoglyph_char_ratio": "contains a high proportion of characters that can visually resemble other letters/digits",
    "has_at_symbol": "contains an '@' symbol, which can hide the real destination",
    "has_double_slash_redirect": "contains a suspicious '//' redirect pattern in the path",
    "is_known_shortener": "uses a URL-shortening service, which hides the real destination",
    "brand_token_in_subdomain_or_path": "mentions a well-known brand name outside of its real domain",
    "url_length": "is unusually long",
    "hostname_length": "has an unusually long hostname",
    "hyphen_count": "contains an unusually high number of hyphens",
    "subdomain_count": "has an unusually high number of subdomains",
    "entropy": "has unusually random-looking characters",
    "has_port": "specifies a non-standard port number",
    "uses_https": "does not use HTTPS",
}


def load_artifacts(model_dir: str = MODEL_DIR):
    rf = joblib.load(f"{model_dir}/random_forest.joblib")
    feature_cols = joblib.load(f"{model_dir}/feature_columns.joblib")
    return rf, feature_cols


def explain_prediction(rf, feature_cols, feature_dict, top_n=5):
    """
    Per-URL explanation using the Random Forest's global feature
    importances, filtered down to which of THIS url's flags are
    actually 'switched on': boolean signals that fired, or continuous
    signals that exceed a normal-range threshold -- not just any
    non-zero value, since e.g. hostname_length is >0 for almost every
    URL and would otherwise show up as a "signal" on trustworthy URLs too.
    """
    importances = dict(zip(feature_cols, rf.feature_importances_))
    active_flags = []
    for feat, val in feature_dict.items():
        if feat.startswith(("has_", "is_")) and val == 1:
            active_flags.append((feat, importances.get(feat, 0)))
        elif feat in CONTINUOUS_THRESHOLDS and val > CONTINUOUS_THRESHOLDS[feat]:
            active_flags.append((feat, importances.get(feat, 0) * val))

    active_flags.sort(key=lambda x: x[1], reverse=True)
    return active_flags[:top_n]


def score_url(url: str, model_dir: str = MODEL_DIR) -> dict:
    rf, feature_cols = load_artifacts(model_dir)
    feats = extract_features(url).as_dict()
    X = np.array([[feats[c] for c in feature_cols]])

    phishing_proba = rf.predict_proba(X)[0, 1]
    trust_score = round((1 - phishing_proba) * 100, 1)  # 0 = fully suspicious, 100 = fully trusted

    reasons = explain_prediction(rf, feature_cols, feats)
    explanations = [FRIENDLY_NAMES.get(f, f) for f, _ in reasons if f in FRIENDLY_NAMES]

    return {
        "url": url,
        "trust_score": trust_score,
        "phishing_probability": round(float(phishing_proba), 4),
        "top_signals": explanations,
        "raw_features": feats,
    }


def cli():
    if len(sys.argv) < 2:
        print("Usage: python -m phishing_detector.predict_url <url>")
        sys.exit(1)

    result = score_url(sys.argv[1])
    print(f"\nURL: {result['url']}")
    print(f"Trust score: {result['trust_score']} / 100")
    print(f"Estimated phishing probability: {result['phishing_probability']}")
    print("Top contributing signals:")
    if result["top_signals"]:
        for s in result["top_signals"]:
            print(f"  - {s}")
    else:
        print("  - No strong suspicious signals detected")


if __name__ == "__main__":
    cli()
