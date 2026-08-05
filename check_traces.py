from backend.database import SessionLocal, TraceRecord
db = SessionLocal()
print('Traces in database:', db.query(TraceRecord).count())
db.close()