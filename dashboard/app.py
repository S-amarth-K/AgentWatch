"""
AgentWatch dashboard -- Streamlit UI.

Run with:
    streamlit run dashboard/app.py

Shows: summary stats, a filterable trace table, and a detail view per trace
(full reasoning/tool-call steps, auditor classification, and a human review
control that feeds the feedback loop).
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import streamlit as st

from dashboard.data import load_traces_df, get_full_trace, submit_human_review, get_summary_stats

st.set_page_config(page_title="AgentWatch", layout="wide")

st.title("AgentWatch — Agent Observability Dashboard")

# ---------------------------------------------------------------------------
# Load data (re-fetched on every rerun, e.g. after a filter change or review
# submission -- simplest correct approach for a dashboard this size).
# ---------------------------------------------------------------------------
df = load_traces_df()

if df.empty:
    st.warning("No traces found yet. Run your agent / generate_dataset.py first, "
               "then refresh this page.")
    st.stop()

stats = get_summary_stats(df)

# ---------------------------------------------------------------------------
# Top summary metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total traces", stats["total_traces"])
col2.metric("Error status", stats["error_count"])
col3.metric("Flagged anomalous", stats["flagged_count"])
col4.metric("Avg anomaly score", stats["avg_anomaly_score"])
col5.metric("Reviewed / false-positive rate",
            f"{stats['reviewed_count']} / {stats['false_positive_rate']:.0%}")

st.divider()

# ---------------------------------------------------------------------------
# Charts: status breakdown + failure_mode breakdown
# ---------------------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Traces by status")
    status_counts = df["status"].value_counts()
    st.bar_chart(status_counts)

with chart_col2:
    st.subheader("Audited failure modes")
    failure_counts = df["failure_mode"].dropna().value_counts()
    if failure_counts.empty:
        st.caption("No traces audited yet -- run auditor.run_auditor first.")
    else:
        st.bar_chart(failure_counts)

st.divider()

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
st.subheader("Traces")

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    status_filter = st.multiselect("Status", options=sorted(df["status"].dropna().unique()))
with filter_col2:
    failure_options = sorted(df["failure_mode"].dropna().unique())
    failure_filter = st.multiselect("Failure mode", options=failure_options)
with filter_col3:
    min_anomaly = st.slider("Minimum anomaly score", 0.0, 1.0, 0.0, 0.05)

filtered_df = df.copy()
if status_filter:
    filtered_df = filtered_df[filtered_df["status"].isin(status_filter)]
if failure_filter:
    filtered_df = filtered_df[filtered_df["failure_mode"].isin(failure_filter)]
filtered_df = filtered_df[filtered_df["anomaly_score"].fillna(0) >= min_anomaly]

st.caption(f"Showing {len(filtered_df)} of {len(df)} traces")

display_df = filtered_df[[
    "trace_id", "timestamp", "status", "user_input",
    "num_steps", "anomaly_score", "failure_mode", "reviewed",
]].copy()
display_df["anomaly_score"] = display_df["anomaly_score"].round(3)

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Trace detail + human review
# ---------------------------------------------------------------------------
st.subheader("Inspect a trace")

selected_id = st.selectbox(
    "Select a trace_id to view details",
    options=[""] + filtered_df["trace_id"].tolist(),
)

if selected_id:
    trace = get_full_trace(selected_id)

    if trace is None:
        st.error("Trace not found.")
    else:
        st.write(f"**User input:** {trace.get('user_input')}")
        st.write(f"**Status:** {trace.get('status')} &nbsp;&nbsp; "
                 f"**Total latency:** {trace.get('total_latency_ms')} ms")

        st.markdown("**Steps:**")
        for step in trace.get("steps", []):
            with st.expander(
                f"Step {step['step_number']}: {step.get('tool_called') or 'final response'} "
                f"({step.get('step_status')})"
            ):
                st.write(f"**Reasoning:** {step.get('decision_rationale')}")
                if step.get("tool_called"):
                    st.write(f"**Tool called:** `{step['tool_called']}`")
                    st.json(step.get("tool_args"))
                    st.write("**Tool result:**")
                    st.json(step.get("tool_result"))
                st.caption(f"Latency: {step.get('latency_ms')} ms")

        st.markdown(f"**Final response:** {trace.get('final_response')}")

        classification = trace.get("auditor_classification")
        if classification:
            st.markdown("---")
            st.markdown("**Auditor classification**")
            st.write(f"Failure mode: `{classification.get('failure_mode')}`")
            st.write(f"Explanation: {classification.get('explanation')}")

            existing_review = trace.get("human_review") or {}
            if existing_review.get("confirmed") is not None:
                verdict = "✅ Confirmed" if existing_review["confirmed"] else "❌ Rejected (false positive)"
                st.info(f"Already reviewed: {verdict}"
                        + (f" — {existing_review.get('reviewer_note')}" if existing_review.get("reviewer_note") else ""))
            else:
                st.markdown("**Human review** (feeds the feedback loop)")
                note = st.text_input("Reviewer note (optional)", key=f"note_{selected_id}")
                rcol1, rcol2 = st.columns(2)
                if rcol1.button("✅ Confirm — this is a real issue", key=f"confirm_{selected_id}"):
                    submit_human_review(selected_id, True, note)
                    st.rerun()
                if rcol2.button("❌ Reject — false positive", key=f"reject_{selected_id}"):
                    submit_human_review(selected_id, False, note)
                    st.rerun()
        else:
            st.caption("Not yet audited by the LLM auditor.")