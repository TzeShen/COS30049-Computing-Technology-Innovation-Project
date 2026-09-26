"""
app.py

Checkie -- the web frontend for the offline phishing URL trust-score
tool, matching the provided design (dark theme, gradient "Checkie"
logo, Trust Score card, "Why this result was given" signal cards, and
a Recent URL history table).

Everything here is Python: Flask routes render Jinja2 templates
server-side, form submissions are plain HTML POSTs, and "recent URLs"
history is stored server-side as JSON (see history.py) rather than in
browser localStorage. There is no JavaScript anywhere in this app.

Fully offline by design: the server only ever reads the pasted URL's
text via feature_extraction.py -- it never fetches the URL itself,
matching the "Checkie analyses URL structure locally... without
opening the webpage" promise shown in the UI.

Run from the repo root, AFTER the training pipeline has produced
models/ (see README):
    python -m phishing_detector.app
Then open http://127.0.0.1:5000
"""

from __future__ import annotations

import os

from flask import Flask, redirect, render_template, request, url_for

from . import history
from .feature_extraction import extract_features
from .predict_url import CONTINUOUS_THRESHOLDS, load_artifacts

app = Flask(__name__)

# Load the trained model once at startup rather than per-request.
# If training hasn't been run yet, _MODEL stays None and the analyse
# form shows a clear, actionable error instead of a stack trace.
_MODEL = None
_FEATURE_COLS = None
_LOAD_ERROR: str | None = None

try:
    _MODEL, _FEATURE_COLS = load_artifacts()
except FileNotFoundError:
    _LOAD_ERROR = (
        "No trained model found. Run the training pipeline first: "
        "python -m phishing_detector.data_prep && "
        "python -m phishing_detector.train_classification"
    )


def _status_from_score(trust_score: float) -> str:
    if trust_score >= 70:
        return "Safe"
    if trust_score >= 40:
        return "Risky"
    return "Dangerous"


STATUS_DESCRIPTIONS = {
    "Safe": "This URL is mostly safe",
    "Risky": "This URL has some suspicious signals",
    "Dangerous": "This URL shows strong signs of phishing",
}


def _build_cards(raw_features: dict) -> list[dict]:
    """
    Maps the model's full feature vector down to the four signal cards
    shown in the UI (IP Address, URL Length, Suspicious TLD, Subdomain
    Count). Thresholds are shared with predict_url.py's CLI explanation
    (CONTINUOUS_THRESHOLDS) so the web UI and the command-line tool
    never disagree about what counts as "unusual".
    """
    ip = raw_features["has_ip_address"]
    tld = raw_features["is_suspicious_tld"]
    url_len = raw_features["url_length"]
    subdomains = raw_features["subdomain_count"]
    len_threshold = CONTINUOUS_THRESHOLDS["url_length"]
    sub_threshold = CONTINUOUS_THRESHOLDS["subdomain_count"]

    return [
        {
            "title": "IP Address",
            "flagged": bool(ip),
            "detail": "Detected" if ip else "Not Detected",
        },
        {
            "title": "URL Length",
            "flagged": url_len > len_threshold,
            "detail": f"{'Long' if url_len > len_threshold else 'Normal'} ({url_len} characters)",
        },
        {
            "title": "Suspicious TLD",
            "flagged": bool(tld),
            "detail": "Detected" if tld else "Not Detected",
        },
        {
            "title": "Subdomain Count",
            "flagged": subdomains >= sub_threshold,
            "detail": (
                f"{'High' if subdomains >= sub_threshold else 'Low' if subdomains <= 1 else 'Moderate'}"
                f" ({subdomains})"
            ),
        },
    ]


@app.get("/")
def index():
    return render_template(
        "index.html",
        active_page="home",
        recent=history.list_recent(10),
        error=request.args.get("error"),
    )


@app.post("/analyse")
def analyse():
    url = (request.form.get("url") or "").strip()
    if not url:
        return redirect(url_for("index"))

    if _MODEL is None:
        return redirect(url_for("index", error=_LOAD_ERROR))

    feats = extract_features(url).as_dict()
    phishing_proba = float(_MODEL.predict_proba([[feats[c] for c in _FEATURE_COLS]])[0, 1])
    trust_score = round((1 - phishing_proba) * 100, 1)
    status = _status_from_score(trust_score)
    cards = _build_cards(feats)

    check_id = history.add_entry(
        url=url,
        trust_score=trust_score,
        risk_percent=round(phishing_proba * 100, 1),
        status=status,
        reasons=cards,
    )
    return redirect(url_for("result", check_id=check_id))


@app.get("/result/<int:check_id>")
def result(check_id: int):
    entry = history.get_entry(check_id)
    if entry is None:
        return redirect(url_for("index"))

    return render_template(
        "result.html",
        active_page="home",
        url=entry["url"],
        trust_score=entry["trust_score"],
        status=entry["status"],
        status_desc=STATUS_DESCRIPTIONS.get(entry["status"], ""),
        reasons=entry["reasons"],
    )


@app.get("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html", active_page="how")


def cli():
    if _LOAD_ERROR:
        print(f"WARNING: {_LOAD_ERROR}")
    app.run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    cli()
