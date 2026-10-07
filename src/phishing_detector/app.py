"""Checkie's local Flask interface for the saved PhiUSIIL EBM.

Run from the repository root with python -m phishing_detector.app.
The CLI and website share one prediction function and the frozen 0.5 threshold.
URL analysis is entirely local; the page and its assets need no remote service.
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, redirect, render_template, request, url_for

from . import history
from .predict_url import (
    MAX_URL_LENGTH,
    MODEL_DIR,
    MODEL_NAME,
    PROJECT_ROOT,
    artifact_signature,
    load_artifacts,
    score_url,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024

_MODEL = None
_FEATURE_COLS = None
_LOAD_ERROR = None

try:
    _MODEL, _FEATURE_COLS = load_artifacts()
    # Give each model/extractor combination its own history. The old demo
    # history remains on disk and is never presented as an EBM prediction.
    signature = artifact_signature()[:12]
    history_base = Path(os.environ.get("CHECKIE_HISTORY_PATH", str(PROJECT_ROOT / "data" / "history.json")))
    history.HISTORY_PATH = str(history_base.with_name(
        f"{history_base.stem}_phiusiil_ebm_{signature}{history_base.suffix or '.json'}"
    ))
except (OSError, ImportError, ValueError, AttributeError) as error:
    _MODEL = None
    _LOAD_ERROR = (
        "Could not load the saved PhiUSIIL EBM. Check ebm.joblib and "
        f"feature_columns.joblib in {MODEL_DIR}, and use your training environment. "
        f"Details: {error}"
    )


def _predict(url: str) -> dict:
    return score_url(url, model=_MODEL, feature_cols=_FEATURE_COLS)


def _render_home(error=None, entered_url="", status_code=200):
    return render_template(
        "index.html",
        active_page="home",
        recent=history.list_recent(10) if _MODEL is not None else [],
        error=error or _LOAD_ERROR,
        entered_url=entered_url,
        max_url_length=MAX_URL_LENGTH,
    ), status_code


@app.get("/")
def index():
    return _render_home()


@app.post("/analyse")
def analyse():
    if _MODEL is None:
        return _render_home(status_code=503)
    url = request.form.get("url", "")
    try:
        prediction = _predict(url)
    except ValueError as error:
        return _render_home(error=str(error), entered_url=url[:MAX_URL_LENGTH], status_code=400)

    try:
        check_id = history.add_entry(
            url=prediction["url"],
            trust_score=prediction["trust_score"],
            risk_percent=prediction["risk_percent"],
            status=prediction["status"],
            reasons=prediction["indicators"],
        )
    except OSError:
        return _render_home(
            error="The result could not be saved. Check that the data folder is writable.",
            entered_url=prediction["url"],
            status_code=500,
        )
    return redirect(url_for("result", check_id=check_id))


@app.get("/result/<int:check_id>")
def result(check_id: int):
    if _MODEL is None:
        return _render_home(status_code=503)
    entry = history.get_entry(check_id)
    if entry is None:
        return redirect(url_for("index"))
    # Recreate contributions using the same frozen model that owns this history.
    # No data is fitted, and an input URL is never fetched or made into a link.
    try:
        prediction = _predict(entry["url"])
    except ValueError as error:
        return _render_home(error=str(error), status_code=400)
    return render_template("result.html", active_page="home", **prediction)


@app.get("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html", active_page="how", model_name=MODEL_NAME)


@app.errorhandler(413)
def request_too_large(_error):
    return _render_home(error="The submitted form is too large. Paste a single URL.", status_code=413)


def cli():
    if _LOAD_ERROR:
        print(f"WARNING: {_LOAD_ERROR}")
    else:
        print(f"Loaded {MODEL_NAME} with {len(_FEATURE_COLS)} features and threshold 0.5.")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    cli()
