"""
Shared, persisted anomaly-flagging threshold.

Previously this was hardcoded separately in auditor/run_auditor.py AND
dashboard/data.py (0.6 in both places) -- meaning recalibrating one would
silently leave the other stale. This module is the single source of truth:
the feedback loop writes to it, everything else reads from it.
"""

from __future__ import annotations

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "threshold_config.json")
DEFAULT_THRESHOLD = 0.6


def load_threshold() -> float:
    """Returns the current flagging threshold, or the default if never calibrated."""
    if not os.path.exists(CONFIG_PATH):
        return DEFAULT_THRESHOLD
    with open(CONFIG_PATH, "r") as f:
        data = json.load(f)
    return float(data.get("threshold", DEFAULT_THRESHOLD))


def save_threshold(value: float, metadata: dict | None = None) -> None:
    """Persists a new threshold, along with optional metadata about the recalibration run."""
    payload = {"threshold": value}
    if metadata:
        payload["metadata"] = metadata
    with open(CONFIG_PATH, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    print(f"Current threshold: {load_threshold()}")