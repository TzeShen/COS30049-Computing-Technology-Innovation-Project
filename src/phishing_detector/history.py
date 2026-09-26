"""
history.py

Server-side storage for the "recent URLs" table on the Checkie
homepage. Each analysed URL is stored on disk as a JSON record with an
incrementing id, so a result page can be revisited via
/result/<id> (e.g. by clicking a row in the history table) without
re-running the model.

Stored as a plain JSON file under data/ -- no database dependency, and
consistent with the rest of the repo's "regenerated, not committed"
data/ convention (see .gitignore).

Kept as pure Python + the standard library json module: no client-side
storage, no JavaScript.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

HISTORY_PATH = os.environ.get("CHECKIE_HISTORY_PATH", "data/history.json")
MAX_ENTRIES = 100


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _load_all() -> list[dict[str, Any]]:
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_all(entries: list[dict[str, Any]]) -> None:
    _ensure_parent(HISTORY_PATH)
    with open(HISTORY_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def add_entry(
    url: str,
    trust_score: float,
    risk_percent: float,
    status: str,
    reasons: list[dict[str, Any]],
) -> int:
    """Persists one analysed URL and returns its new id."""
    entries = _load_all()
    next_id = (max((e["id"] for e in entries), default=0)) + 1
    entry = {
        "id": next_id,
        "url": url,
        "trust_score": trust_score,
        "risk_percent": risk_percent,
        "status": status,
        "reasons": reasons,
        "checked_at": datetime.now().strftime("%b - %d - %Y"),
    }
    entries.insert(0, entry)
    _save_all(entries[:MAX_ENTRIES])
    return next_id


def get_entry(check_id: int) -> dict[str, Any] | None:
    for entry in _load_all():
        if entry["id"] == check_id:
            return entry
    return None


def list_recent(limit: int = 10) -> list[dict[str, Any]]:
    # Entries are inserted most-recent-first, so no re-sort is needed
    # for the default view.
    return _load_all()[:limit]
