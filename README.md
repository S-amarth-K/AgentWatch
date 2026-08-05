# AgentWatch

**An ML-LLM Hybrid Observability and Governance Framework for Autonomous AI Agents**

AgentWatch monitors the execution of an autonomous AI agent, automatically flags anomalous behavior using machine learning, uses an LLM-based auditor to classify *why* a flagged run looks wrong, and recalibrates its own detection accuracy over time using human feedback.

Most AI agent deployments today have no systematic way to catch failures like incorrect tool usage, hallucinated success claims, or infinite loops until a user reports a problem. AgentWatch is a working prototype of the missing observability layer for agentic systems — combining unsupervised anomaly detection, LLM-based root-cause classification, and a closed human-feedback loop.

---

## Architecture

```
Scripted scenarios ──▶ Support agent (Gemini 2.5 Flash, tool-calling)
                              │
                              ▼
                    Execution trace (schema-validated)
                              │
                              ▼
                 FastAPI ingestion API ──▶ SQLite database
                              │
                              ▼
              ML anomaly detection (Isolation Forest)
                     writes anomaly_score
                              │
                              ▼
             LLM auditor classifies flagged traces
             (failure_mode + human-readable explanation)
                              │
                              ▼
         Streamlit dashboard (charts, trace inspection,
                human review: confirm / reject)
                              │
                              ▼
        Feedback loop recalibrates the flagging threshold
              using confirmed/rejected reviews
```

## Key components

| Component | Technology | Purpose |
|---|---|---|
| Agent | Gemini 2.5 Flash, hand-rolled tool-calling loop | Simulates a support agent with `check_order_status`, `issue_refund`, `escalate_to_human` tools |
| Trace schema | Pydantic + JSON Schema | Structured record of every reasoning step, tool call, and result |
| Storage | FastAPI + SQLite (SQLAlchemy) | Ingests and persists traces; exposes REST endpoints |
| Anomaly detection | scikit-learn (Isolation Forest) | Unsupervised detection of statistically unusual traces from 12 engineered features |
| LLM auditor | Gemini 2.5 Flash (+ rule-based mock mode) | Classifies flagged traces into a failure-mode taxonomy with an explanation |
| Dashboard | Streamlit | Visualizes traces, scores, and failure modes; captures human review |
| Feedback loop | Custom (Youden's J threshold search) | Recalibrates the anomaly-flagging threshold using accumulated human reviews |

## Failure-mode taxonomy

The auditor classifies every flagged trace into one of:

- `wrong_tool_call` — inappropriate tool or invalid arguments given the context
- `hallucinated_success` — final response claims something not supported by tool results
- `infinite_loop` — same tool called repeatedly without progress
- `unauthorized_action` — action taken without sufficient justification (e.g. double refund)
- `low_confidence_answered_anyway` — answered definitively despite ambiguity
- `policy_violation` — violates an explicit system policy
- `infrastructure_error` — failure caused by API/network issues, not agent reasoning
- `none` — no issue detected

## Results

On a locally generated evaluation set:

- The anomaly detection model, after correcting for latency-scale distortion (log-transform) and switching from `contamination="auto"` to a fixed estimate, produced a stable ~15-20% flag rate instead of an initial ~46% over-flagging rate.
- The LLM auditor correctly distinguished genuine reasoning failures (loops, wrong tool calls, policy violations) from **infrastructure-level failures** (e.g. API rate limits) — a distinction the initial heuristic classifier missed, caught during testing, and fixed by adding a dedicated `infrastructure_error` category.
- After one round of human review and recalibration, the **false-positive rate dropped from 100% to 50% on the reviewed set, while recall (true-positive rate) remained at 100%** — demonstrating that detection accuracy improves with feedback without missing genuine issues.

## Project structure

```
AgentWatch/
├── agents/
│   ├── tools.py              # mock tool implementations (order lookup, refund, escalation)
│   └── support_agent.py      # tool-calling loop (mock + real Gemini modes)
├── auditor/
│   ├── taxonomy.py           # failure-mode definitions
│   ├── auditor.py            # LLM/heuristic classification logic
│   └── run_auditor.py        # batch script: audits all flagged traces
├── backend/
│   ├── models/
│   │   └── trace_models.py   # Pydantic schema (single source of truth)
│   ├── database.py           # SQLAlchemy models + session handling
│   └── main.py                # FastAPI app (ingestion, retrieval, review endpoints)
├── dashboard/
│   ├── data.py                # data access layer (testable independent of UI)
│   └── app.py                 # Streamlit UI
├── ml/
│   ├── feature_extraction.py  # trace → numeric feature vector
│   ├── train_anomaly_model.py # Isolation Forest training + scoring
│   ├── threshold_config.py    # shared, persisted flagging threshold
│   ├── feedback_loop.py       # recalibration using human review data
│   └── models/                # saved model artifacts (generated)
├── scripts/
│   └── generate_dataset.py    # batch trace generator (randomized scenarios)
├── schema.json                 # formal JSON Schema for the trace format
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/AgentWatch.git
cd AgentWatch
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
```

## Running the full pipeline

```bash
# Terminal 1 — API server (leave running)
uvicorn backend.main:app --reload

# Terminal 2 — generate trace data
python scripts/generate_dataset.py        # repeat a few times for more data

# Score anomalies
python ml/train_anomaly_model.py

# Classify flagged traces
python -m auditor.run_auditor

# Launch the dashboard
streamlit run dashboard/app.py

# After reviewing traces in the dashboard (confirm/reject), recalibrate:
python ml/feedback_loop.py
```

`MOCK_MODE` (in `agents/support_agent.py` and `auditor/auditor.py`) toggles between free, instant scripted responses and real Gemini API calls — useful for building large datasets without consuming API quota.

## Known limitations

- The current dataset is generated from hand-written scenario templates, not real production traffic.
- Anomaly detection is unsupervised and has no ground truth until human review accumulates; small datasets can produce unstable flag rates.
- The dashboard reads/writes directly against the database rather than exclusively through the API layer, a deliberate simplification for a single-operator observability tool.

## Future work

- Faithfulness/hallucination scoring layer (NLI-based) comparing agent responses against tool outputs
- Larger-scale evaluation with a bigger reviewed dataset
- Multi-agent support (currently a single support-agent scenario)

## License

Academic project — final-year submission.