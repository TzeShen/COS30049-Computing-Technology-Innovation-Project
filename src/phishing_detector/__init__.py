"""
phishing_detector

Offline URL-based phishing signal extraction, classification,
clustering, and scoring.

Public API:
    extract_features(url) -> URLFeatures
    score_url(url)         -> dict (trust score + explanation)
"""

from .feature_extraction import URLFeatures, extract_features, extract_features_batch

__all__ = [
    "URLFeatures",
    "extract_features",
    "extract_features_batch",
    "score_url",
]

__version__ = "0.1.0"


def __getattr__(name):
    # Lazily import score_url rather than importing predict_url eagerly
    # above: predict_url is also meant to be run directly as a module
    # (`python -m phishing_detector.predict_url <url>`), and eagerly
    # importing it here as well trips Python's "module already in
    # sys.modules" runtime warning when doing so. Deferring the import
    # until score_url is actually accessed avoids that entirely while
    # still supporting `from phishing_detector import score_url`.
    if name == "score_url":
        from .predict_url import score_url
        return score_url
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
