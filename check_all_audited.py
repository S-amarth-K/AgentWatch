import json

from backend.database import SessionLocal, TraceRecord


def main():
    db = SessionLocal()

    try:
        records = (
            db.query(TraceRecord)
            .filter(TraceRecord.anomaly_score >= 0.6)
            .order_by(TraceRecord.anomaly_score.desc())
            .all()
        )

        if not records:
            print("No traces found with anomaly_score >= 0.6")
            return

        print(f"\n=== {len(records)} Audited Traces ===\n")

        for record in records:
            trace = json.loads(record.full_trace_json)
            classification = trace.get("auditor_classification") or {}

            print(f"--- {record.trace_id} (anomaly_score={record.anomaly_score:.2f}) ---")
            print(f"  user_input: {trace.get('user_input')}")
            print(f"  failure_mode: {classification.get('failure_mode')}")
            print(f"  explanation: {classification.get('explanation')}")
            print()

    finally:
        db.close()


if __name__ == "__main__":
    main()