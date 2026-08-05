"""
Data access layer for the dashboard -- kept separate from the Streamlit UI
code so these functions can be tested directly without needing a running
Streamlit app.

Reads/writes directly against the SQLite database (same pattern as
ml/train_anomaly_model.py and auditor/run_auditor.py), rather than going
through the FastAPI layer. This is a deliberate choice: an internal
observability dashboard reading straight from the store is a normal,
legitimate pattern (avoids requiring both uvicorn AND streamlit running
just to view data), though it does mean the dashboard bypasses the API's
validation on writes -- worth a one-line mention in your report as a
known architectural trade-off.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from backend.database import SessionLocal, TraceRecord
from ml.threshold_config import load_threshold


def load_traces_df() -> pd.DataFrame:
    """Loads all traces as a flat DataFrame for table display and charts."""
    db = SessionLocal()
    try:
        records = db.query(TraceRecord).order_by(TraceRecord.timestamp.desc()).all()
        rows = []
        for r in records:
            full_trace = json.loads(r.full_trace_json)
            classification = full_trace.get("auditor_classification") or {}
            review = full_trace.get("human_review") or {}
            rows.append({
                "trace_id": r.trace_id,
                "timestamp": r.timestamp,
                "agent_name": r.agent_name,
                "status": r.status,
                "user_input": full_trace.get("user_input", "")[:80],
                "num_steps": len(full_trace.get("steps", [])),
                "total_latency_ms": r.total_latency_ms,
                "anomaly_score": r.anomaly_score,
                "failure_mode": classification.get("failure_mode"),
                "reviewed": review.get("confirmed") is not None,
                "review_confirmed": review.get("confirmed"),
            })
        return pd.DataFrame(rows)
    finally:
        db.close()


def get_full_trace(trace_id: str) -> dict | None:
    """Fetches the complete trace dict (all steps, reasoning, tool calls) by ID."""
    db = SessionLocal()
    try:
        record = db.get(TraceRecord, trace_id)
        if record is None:
            return None
        return json.loads(record.full_trace_json)
    finally:
        db.close()


def submit_human_review(trace_id: str, confirmed: bool, reviewer_note: str = "") -> bool:
    """
    Records a human reviewer's decision on a trace's auditor classification.
    This is the exact input the feedback loop (recalibration) will consume.
    """
    db = SessionLocal()
    try:
        record = db.get(TraceRecord, trace_id)
        if record is None:
            return False
        full_trace = json.loads(record.full_trace_json)
        full_trace["human_review"] = {"confirmed": confirmed, "reviewer_note": reviewer_note}
        record.full_trace_json = json.dumps(full_trace)
        db.commit()
        return True
    finally:
        db.close()


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Computes headline numbers for the dashboard's top metrics row."""
    if df.empty:
        return {
            "total_traces": 0, "error_count": 0, "flagged_count": 0,
            "avg_anomaly_score": 0.0, "reviewed_count": 0, "false_positive_rate": 0.0,
        }

    total = len(df)
    error_count = int((df["status"] == "error").sum())
    threshold = load_threshold()
    flagged_count = int((df["anomaly_score"] >= threshold).sum())
    avg_score = float(df["anomaly_score"].dropna().mean()) if df["anomaly_score"].notna().any() else 0.0
    reviewed_count = int(df["reviewed"].sum())

    reviewed_df = df[df["reviewed"] == True]  # noqa: E712
    false_positive_rate = (
        float((reviewed_df["review_confirmed"] == False).mean())  # noqa: E712
        if len(reviewed_df) > 0 else 0.0
    )

    return {
        "total_traces": total,
        "error_count": error_count,
        "flagged_count": flagged_count,
        "avg_anomaly_score": round(avg_score, 3),
        "reviewed_count": reviewed_count,
        "false_positive_rate": round(false_positive_rate, 3),
    }