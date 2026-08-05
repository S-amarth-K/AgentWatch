"""
SQLite storage layer for agent traces, using SQLAlchemy.

Design choice: we store the FULL trace as JSON (so nothing is ever lost or
needs a schema migration when you add new fields later), PLUS a handful of
commonly-filtered fields as real indexed columns (status, agent_name,
timestamp, anomaly_score, faithfulness_score) so the dashboard and ML layer
can query efficiently without parsing every JSON blob on every request.
"""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import Column, String, Float, DateTime, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./traces.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TraceRecord(Base):
    __tablename__ = "traces"

    trace_id = Column(String, primary_key=True, index=True)
    session_id = Column(String, index=True)
    agent_name = Column(String, index=True)
    agent_version = Column(String)
    timestamp = Column(DateTime, index=True)
    status = Column(String, index=True)
    anomaly_score = Column(Float, nullable=True)
    faithfulness_score = Column(Float, nullable=True)
    total_latency_ms = Column(Float, nullable=True)

    # Full trace payload (all steps, reasoning, tool calls/results, auditor
    # classification, human review, etc.) stored as a JSON string.
    full_trace_json = Column(Text)


def init_db() -> None:
    """Creates the traces table if it doesn't already exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a DB session and ensures it's closed after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def trace_record_to_dict(record: TraceRecord) -> dict:
    """Converts a stored TraceRecord back into the full trace dict."""
    return json.loads(record.full_trace_json)