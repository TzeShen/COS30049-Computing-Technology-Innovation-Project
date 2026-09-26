"""
tests/test_app.py

Route-level tests for the Checkie Flask app, using Flask's test
client (no real server/socket needed). Requires a trained model in
models/ -- run the training pipeline first (see README) before running
this test file. Uses a temporary history file so it never touches
data/history.json.

Run from the repo root:
    pytest tests/test_app.py
"""

import os

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Redirect history storage to a scratch file so tests never read or
    # write the real data/history.json.
    monkeypatch.setenv("CHECKIE_HISTORY_PATH", str(tmp_path / "history.json"))

    from phishing_detector import history
    history.HISTORY_PATH = os.environ["CHECKIE_HISTORY_PATH"]

    from phishing_detector.app import app as flask_app
    flask_app.testing = True
    return flask_app.test_client()


def _skip_if_no_model():
    from phishing_detector.app import _MODEL
    if _MODEL is None:
        pytest.skip("No trained model in models/ -- run the training pipeline first.")


def test_homepage_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Phishing URL Detector" in resp.data


def test_how_it_works_loads(client):
    resp = client.get("/how-it-works")
    assert resp.status_code == 200
    assert b"How Checkie works" in resp.data


def test_analyse_redirects_to_result(client):
    _skip_if_no_model()
    resp = client.post("/analyse", data={"url": "http://192.168.1.1/paypal/login.php"})
    assert resp.status_code == 302
    assert resp.headers["Location"].startswith("/result/")


def test_result_page_shows_trust_score_and_reasons(client):
    _skip_if_no_model()
    resp = client.post("/analyse", data={"url": "http://192.168.1.1/paypal/login.php"})
    result_resp = client.get(resp.headers["Location"])
    assert result_resp.status_code == 200
    assert b"Trust Score" in result_resp.data
    assert b"IP Address" in result_resp.data
    assert b"Detected" in result_resp.data  # this URL uses a raw IP host


def test_analysed_url_appears_in_history(client):
    _skip_if_no_model()
    client.post("/analyse", data={"url": "http://192.168.1.1/paypal/login.php"})
    resp = client.get("/")
    assert b"192.168.1.1" in resp.data


def test_empty_url_redirects_home_without_crashing(client):
    resp = client.post("/analyse", data={"url": ""})
    assert resp.status_code == 302
    assert resp.headers["Location"] == "/"
