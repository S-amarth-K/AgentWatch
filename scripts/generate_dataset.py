"""
Batch trace generator (v2) — larger, randomized scenario pool.

Each run randomly samples N scenarios (with repetition allowed across runs,
and slight input phrasing variation), so running this multiple times keeps
adding genuinely varied traces instead of repeating the same 8 every time.

IMPORTANT: start the API server first in another terminal:
    uvicorn backend.main:app --reload
Then run this script:
    python scripts/generate_dataset.py
"""

from __future__ import annotations

import os
import random
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agents.support_agent import run_agent, send_trace_to_api  # noqa: E402

# A pool of (label, template) scenarios. Templates get lightly varied so
# repeated runs don't produce byte-identical user_input strings either.
SCENARIO_POOL = [
    ("normal_refund_valid", [
        "I want a refund for order #4521, it never arrived.",
        "Refund order #4521 please, it's very late.",
        "Order #4521 hasn't shown up, can I get a refund?",
    ]),
    ("normal_status_check", [
        "Can you tell me the status of order #1001?",
        "What's happening with order #1001?",
        "Where is my order #1001?",
    ]),
    ("normal_delivered_no_issue", [
        "When will order #2002 arrive?",
        "Status update on order #2002 please.",
    ]),
    ("edge_already_refunded", [
        "I'd like a refund for order #3003, it's broken.",
        "Order #3003 arrived damaged, refund please.",
    ]),
    ("edge_unknown_order", [
        "Please refund order #9999, I never got it.",
        "Refund order #9999, it's missing.",
    ]),
    ("edge_ambiguous_request", [
        "My order is messed up, fix it please.",
        "Something's wrong with my order, help.",
    ]),
    ("edge_loop_trigger", [
        "Can you double check order #4521, loop through it carefully?",
        "Please loop-verify order #4521 status thoroughly.",
    ]),
    ("normal_refund_valid_repeat", [
        "Refund order #4521, it's late.",
    ]),
]

SAMPLE_SIZE = 12  # traces generated per run


def main():
    random.seed()  # true randomness each run, not reproducible on purpose

    sampled = []
    for _ in range(SAMPLE_SIZE):
        label, templates = random.choice(SCENARIO_POOL)
        user_input = random.choice(templates)
        sampled.append((label, user_input))

    print(f"Running {SAMPLE_SIZE} randomly sampled scenarios...\n")
    stored_count = 0

    for label, user_input in sampled:
        print(f"--- [{label}] \"{user_input}\" ---")
        trace = run_agent(user_input)
        print(f"    status={trace.status}, steps={len(trace.steps)}, latency_ms={trace.total_latency_ms}")

        success = send_trace_to_api(trace)
        if success:
            stored_count += 1

        print()
        time.sleep(0.3)

    print(f"✅ Done. {stored_count}/{SAMPLE_SIZE} traces stored successfully.")
    print("   Go to http://127.0.0.1:8000/docs -> GET /traces to view them.")
    print("   Run this script again for another batch of randomized scenarios.")


if __name__ == "__main__":
    main()