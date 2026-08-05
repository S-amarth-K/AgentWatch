"""
FastAPI ingestion API for agent traces.

Endpoints:
    POST /traces              -> store a new trace
    GET  /traces               -> list traces (optionally filtered)
    GET  /traces/{trace_id}     -> fetch one full trace
    PATCH /traces/{trace_id}/review -> record a human review decision (used
                                        later by the feedback loop)

Run with:  uvicorn backend.main:app --reload
Docs at:   http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import sys
import os
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import init_db, get_db, TraceRecord, trace_record_to_dict
from backend.models.trace_models import AgentTrace, HumanReview

app = FastAPI(title="Agent Observability API", version="0.1.0")


@app.on_event("startup")
def on_startup():
    init_db()


@app.post("/traces", status_code=201)
def create_trace(trace: AgentTrace, db: Session = Depends(get_db)):
    """Store a new agent execution trace."""
    record = TraceRecord(
        trace_id=str(trace.trace_id),
        session_id=str(trace.session_id),
        agent_name=trace.agent_name,
        agent_version=trace.agent_version,
        timestamp=trace.timestamp,
        status=trace.status,
        anomaly_score=trace.anomaly_score,
        faithfulness_score=trace.faithfulness_score,
        total_latency_ms=trace.total_latency_ms,
        full_trace_json=trace.model_dump_json(),
    )
    existing = db.get(TraceRecord, str(trace.trace_id))
    if existing:
        raise HTTPException(status_code=409, detail=f"Trace {trace.trace_id} already exists.")

    db.add(record)
    db.commit()
    return {"trace_id": trace.trace_id, "stored": True}


@app.get("/traces")
def list_traces(
    status: Optional[str] = None,
    agent_name: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """List traces, optionally filtered by status or agent_name."""
    query = db.query(TraceRecord)
    if status:
        query = query.filter(TraceRecord.status == status)
    if agent_name:
        query = query.filter(TraceRecord.agent_name == agent_name)

    records = query.order_by(TraceRecord.timestamp.desc()).limit(limit).all()
    return [
        {
            "trace_id": r.trace_id,
            "agent_name": r.agent_name,
            "timestamp": r.timestamp,
            "status": r.status,
            "anomaly_score": r.anomaly_score,
            "faithfulness_score": r.faithfulness_score,
            "total_latency_ms": r.total_latency_ms,
        }
        for r in records
    ]


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str, db: Session = Depends(get_db)):
    """Fetch the full trace (all steps, reasoning, tool calls) by ID."""
    record = db.get(TraceRecord, trace_id)
    if not record:
        raise HTTPException(status_code=404, detail="Trace not found.")
    return trace_record_to_dict(record)


@app.patch("/traces/{trace_id}/review")
def submit_human_review(trace_id: str, review: HumanReview, db: Session = Depends(get_db)):
    """
    Record a human reviewer's confirmation/rejection of a flagged trace.
    This is the input the feedback loop (Phase 4) will use to recalibrate
    detection thresholds.
    """
    record = db.get(TraceRecord, trace_id)
    if not record:
        raise HTTPException(status_code=404, detail="Trace not found.")

    full_trace = json.loads(record.full_trace_json)
    full_trace["human_review"] = review.model_dump()
    record.full_trace_json = json.dumps(full_trace)
    db.commit()
    return {"trace_id": trace_id, "review_recorded": True}
