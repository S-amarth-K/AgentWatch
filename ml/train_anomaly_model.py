"""
Trains an Isolation Forest anomaly detection model on stored traces, and
writes the resulting anomaly_score back into the database for each trace.

Isolation Forest works by randomly partitioning the feature space; points
that get isolated in fewer splits (i.e. sit apart from the bulk of "normal"
traces) get a higher anomaly score. It's unsupervised -- it does NOT need
labels -- which matters here because you won't have a large hand-labeled
dataset early on. As you accumulate confirmed human reviews later (Phase 4,
the feedback loop), those labels can be used to validate/calibrate this
model's threshold, but the model itself doesn't require them to train.

Run with:
    python ml/train_anomaly_model.py

Requires the API server NOT to be running exclusively -- this script talks
directly to the SQLite database, not through the API, so it's safe to run
even while uvicorn is up.
"""

from __future__ import annotations

import math
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from backend.database import SessionLocal, TraceRecord
from ml.feature_extraction import trace_to_features, FEATURE_NAMES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "anomaly_model.joblib")


def load_traces_as_dataframe() -> tuple[pd.DataFrame, list[str]]:
    """Reads all stored traces from the DB and extracts features into a DataFrame."""
    db = SessionLocal()
    try:
        records = db.query(TraceRecord).all()
        trace_ids = []
        feature_rows = []
        for record in records:
            trace_dict = json.loads(record.full_trace_json)
            feature_rows.append(trace_to_features(trace_dict))
            trace_ids.append(record.trace_id)
        df = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
        return df, trace_ids
    finally:
        db.close()


def train_and_score() -> None:
    df, trace_ids = load_traces_as_dataframe()

    if len(df) < 5:
        print(f"⚠️  Only {len(df)} traces found. Isolation Forest needs more data to be "
              f"meaningful (aim for 20+). Generate more traces first via generate_dataset.py.")
        if len(df) == 0:
            return

    print(f"Training on {len(df)} traces with {len(FEATURE_NAMES)} features...")

    # Log-transform latency features. Without this, a handful of extreme
    # outliers (e.g. a 170-second API rate-limit stall vs. a normal 50-300ms
    # run -- a ~1000x spread) dominate the split points Isolation Forest
    # chooses, distorting how ALL other points get scored. log1p compresses
    # the extreme values while preserving relative ordering, so genuine
    # latency anomalies are still detected without swamping the model.
    df_transformed = df.copy()
    for col in ["total_latency_ms", "avg_step_latency_ms", "max_step_latency_ms"]:
        df_transformed[col] = df_transformed[col].apply(math.log1p)

    # A fixed contamination estimate is more stable than "auto" on small /
    # mixed-quality datasets, where "auto" can flag an unrealistically large
    # fraction of points. 0.15 is a reasonable starting assumption -- revisit
    # once you have real labeled data from the human feedback loop (Phase 4).
    model = IsolationForest(
        n_estimators=100,
        contamination=0.15,
        random_state=42,
    )
    model.fit(df_transformed)

    # decision_function: higher = more normal, lower/negative = more anomalous.
    # We flip and rescale to a 0-1 "anomaly_score" where 1 = most anomalous,
    # which is more intuitive to read on a dashboard than a raw signed score.
    raw_scores = model.decision_function(df_transformed)
    anomaly_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)

    predictions = model.predict(df_transformed)  # -1 = anomaly, 1 = normal

    # Save the trained model for reuse (e.g. scoring new traces later without
    # retraining from scratch).
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Write scores back into the database.
    db = SessionLocal()
    try:
        flagged_count = 0
        for trace_id, score, pred in zip(trace_ids, anomaly_scores, predictions):
            record = db.get(TraceRecord, trace_id)
            if record is None:
                continue
            record.anomaly_score = float(score)

            full_trace = json.loads(record.full_trace_json)
            full_trace["anomaly_score"] = float(score)
            record.full_trace_json = json.dumps(full_trace)

            if pred == -1:
                flagged_count += 1
        db.commit()
    finally:
        db.close()

    print(f"✅ Scored {len(trace_ids)} traces. {flagged_count} flagged as anomalous by Isolation Forest.")

    # Print a quick summary table for sanity-checking.
    df_summary = df.copy()
    df_summary["anomaly_score"] = anomaly_scores
    df_summary["flagged"] = predictions == -1
    df_summary["trace_id"] = trace_ids
    print("\nTop 5 most anomalous traces:")
    print(
        df_summary.sort_values("anomaly_score", ascending=False)
        .head(5)[["trace_id", "anomaly_score", "flagged", "num_steps", "is_error_status", "max_step_latency_ms"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    train_and_score()