"""
Support agent: hand-rolled tool-calling loop using Gemini 2.5 Flash.

Every reasoning step, tool call, and result is logged into the AgentTrace
schema (backend/models/trace_models.py) as it happens.

MOCK_MODE:
    True  -> no API calls made. Scenario-aware scripted responses are used
             instead, so you can generate a varied labeled dataset quickly
             and for free, without needing network access or burning your
             Gemini quota.
    False -> real calls to Gemini 2.5 Flash via the google-genai SDK.
             Requires GEMINI_API_KEY set in your .env file (loaded via
             python-dotenv).

Run this file directly to execute one agent turn, print the resulting
trace as JSON, and send it to the ingestion API.
"""

from __future__ import annotations

import os
import random
import sys
import time
from uuid import uuid4

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from backend.models.trace_models import AgentTrace, Step, StepStatus, TraceStatus  # noqa: E402
from agents.tools import TOOL_REGISTRY, TOOL_DECLARATIONS  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed yet — fine in MOCK_MODE

MOCK_MODE = True  # flip to False once you've added your real GEMINI_API_KEY
MAX_STEPS = 6

AGENT_NAME = "support_agent"
AGENT_VERSION = "v1"

SYSTEM_PROMPT = """You are a customer support agent. You have access to tools to
check order status, issue refunds, and escalate to a human when needed.
Always check order status before issuing a refund. Never issue a refund twice
for the same order. If you are unsure or the situation is ambiguous, escalate
to a human rather than guessing. Briefly state your reasoning in plain text
before calling a tool."""


# --------------------------------------------------------------------------
# MOCK PATH — scenario-aware scripted responses, no real Content/Part objects needed.
# --------------------------------------------------------------------------

def scripted_plan(user_input: str):
    """
    Returns a list of (tool_name, tool_args, reasoning) steps based on what
    the input actually contains, so mock-mode runs produce realistic,
    varied traces instead of one repeated pattern -- needed for a usable
    anomaly-detection training set.
    """
    ui = user_input.lower()

    if "4521" in ui and "loop" not in ui:
        return [
            ("check_order_status", {"order_id": "4521"}, "Checking order status before refunding."),
            ("issue_refund", {"order_id": "4521", "reason": "not_delivered"}, "Delivery overdue, issuing refund."),
        ]
    if "9999" in ui:
        return [
            ("check_order_status", {"order_id": "9999"}, "Checking order status."),
        ]  # order doesn't exist -> tool returns error -> failure trace
    if "3003" in ui:
        return [
            ("check_order_status", {"order_id": "3003"}, "Checking order status before refunding."),
            ("issue_refund", {"order_id": "3003", "reason": "damaged"}, "Attempting refund."),
        ]  # already refunded -> tool returns error -> failure trace
    if "1001" in ui:
        return [
            ("check_order_status", {"order_id": "1001"}, "Checking order status."),
        ]
    if "2002" in ui:
        return [
            ("check_order_status", {"order_id": "2002"}, "Checking order status."),
        ]
    if "loop" in ui:
        return [
            ("check_order_status", {"order_id": "4521"}, "Checking order status."),
            ("check_order_status", {"order_id": "4521"}, "Checking again to be sure."),
            ("check_order_status", {"order_id": "4521"}, "Checking once more."),
        ]  # repeated identical tool call -> loop signal for anomaly detection
    # ambiguous / no order ID given
    return [
        ("escalate_to_human", {"order_id": "0000", "issue_summary": "Ambiguous request, no order ID provided."}, "Request is ambiguous, escalating to a human."),
    ]


def run_agent_mock(user_input: str) -> tuple[list[Step], str, TraceStatus, str | None]:
    steps: list[Step] = []
    final_response = ""
    status = TraceStatus.success
    error_message = None

    plan = scripted_plan(user_input)
    # ~15% chance of an artificial latency spike on the first step, so the
    # dataset contains realistic latency-anomaly examples too.
    inject_latency_spike = random.random() < 0.15

    for i, (tool_name, tool_args, reasoning) in enumerate(plan, start=1):
        step_start = time.perf_counter()
        if inject_latency_spike and i == 1:
            time.sleep(0.3)

        tool_fn = TOOL_REGISTRY.get(tool_name)
        tool_result = tool_fn(**tool_args) if tool_fn else {"error": f"Unknown tool '{tool_name}'"}
        step_status = StepStatus.failure if "error" in tool_result else StepStatus.success

        steps.append(Step(
            step_number=i,
            decision_rationale=reasoning,
            tool_called=tool_name, tool_args=tool_args, tool_result=tool_result,
            step_status=step_status,
            latency_ms=round((time.perf_counter() - step_start) * 1000, 2),
        ))

        if step_status == StepStatus.failure:
            status = TraceStatus.error
            final_response = f"Something went wrong: {tool_result.get('error')}"
            break
    else:
        final_response = "Your request has been handled successfully."
        status = TraceStatus.success

    return steps, final_response, status, error_message


# --------------------------------------------------------------------------
# REAL PATH — proper Gemini types.Content / types.Part multi-turn handling.
# --------------------------------------------------------------------------

def run_agent_real(user_input: str) -> tuple[list[Step], str, TraceStatus, str | None]:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return [], "", TraceStatus.error, "GEMINI_API_KEY not set. Add it to your .env file."

    # 30-second timeout on the HTTP call itself, so a stalled/rate-limited
    # request fails loudly instead of hanging forever with no error.
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=30_000))
    tools = [types.Tool(function_declarations=TOOL_DECLARATIONS)]
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=tools)

    # Real Gemini conversation history — a list of types.Content objects.
    contents: list = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
    ]

    steps: list[Step] = []
    final_response = ""
    status = TraceStatus.incomplete
    error_message = None

    for step_number in range(1, MAX_STEPS + 1):
        step_start = time.perf_counter()
        print(f"[step {step_number}] calling Gemini API...", flush=True)
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=contents, config=config,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[step {step_number}] API call failed: {exc}", flush=True)
            error_message = str(exc)
            status = TraceStatus.error
            break
        print(f"[step {step_number}] API call returned in {round((time.perf_counter() - step_start) * 1000, 1)}ms", flush=True)

        candidate = response.candidates[0]
        contents.append(candidate.content)  # keep model's turn in history

        # A turn can contain a text part (reasoning) and/or a function_call part.
        reasoning_text = ""
        function_call = None
        for part in candidate.content.parts:
            if getattr(part, "text", None):
                reasoning_text += part.text
            if getattr(part, "function_call", None):
                function_call = part.function_call

        if function_call is None:
            # No tool call -> this is the agent's final answer.
            final_response = reasoning_text or "(model returned an empty response)"
            steps.append(Step(
                step_number=step_number,
                decision_rationale=reasoning_text or "(model produced a direct text response)",
                tool_called=None, tool_args=None, tool_result=None,
                step_status=StepStatus.success,
                latency_ms=round((time.perf_counter() - step_start) * 1000, 2),
            ))
            status = TraceStatus.success
            break

        # Execute the requested tool.
        tool_name = function_call.name
        tool_args = dict(function_call.args)
        tool_fn = TOOL_REGISTRY.get(tool_name)

        if tool_fn is None:
            tool_result = {"error": f"Unknown tool '{tool_name}' requested by model."}
            step_status = StepStatus.failure
        else:
            try:
                tool_result = tool_fn(**tool_args)
                step_status = StepStatus.failure if "error" in tool_result else StepStatus.success
            except Exception as exc:  # noqa: BLE001
                tool_result = {"error": str(exc)}
                step_status = StepStatus.failure

        steps.append(Step(
            step_number=step_number,
            decision_rationale=reasoning_text or "(model provided no explicit reasoning text)",
            tool_called=tool_name, tool_args=tool_args, tool_result=tool_result,
            step_status=step_status,
            latency_ms=round((time.perf_counter() - step_start) * 1000, 2),
        ))

        # Feed the function's result back to the model as the next turn.
        function_response_part = types.Part.from_function_response(
            name=tool_name, response=tool_result,
        )
        contents.append(types.Content(role="tool", parts=[function_response_part]))
    else:
        status = TraceStatus.incomplete
        final_response = "(agent did not produce a final response within max steps)"

    return steps, final_response, status, error_message


# --------------------------------------------------------------------------
# Shared entry point
# --------------------------------------------------------------------------

def run_agent(user_input: str) -> AgentTrace:
    """Runs the tool-calling loop end-to-end and returns a populated AgentTrace."""
    start_time = time.perf_counter()

    if MOCK_MODE:
        steps, final_response, status, error_message = run_agent_mock(user_input)
    else:
        steps, final_response, status, error_message = run_agent_real(user_input)

    total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return AgentTrace(
        session_id=uuid4(),
        agent_name=AGENT_NAME,
        agent_version=AGENT_VERSION,
        user_input=user_input,
        steps=steps,
        final_response=final_response,
        status=status,
        total_latency_ms=total_latency_ms,
        error_message=error_message,
    )


API_BASE_URL = os.environ.get("AGENTWATCH_API_URL", "http://127.0.0.1:8000")


def send_trace_to_api(trace: AgentTrace) -> bool:
    """
    POSTs a trace to the ingestion API. Returns True on success, False on
    failure (e.g. server not running) — never raises, so a dead API server
    doesn't crash agent runs; it just skips storage with a warning.
    """
    import requests

    try:
        response = requests.post(
            f"{API_BASE_URL}/traces",
            data=trace.model_dump_json(),
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        if response.status_code == 201:
            print(f"[stored] trace {trace.trace_id} saved to API.")
            return True
        else:
            print(f"[warning] API returned {response.status_code}: {response.text}")
            return False
    except Exception as exc:  # noqa: BLE001
        print(f"[warning] Could not reach API at {API_BASE_URL}: {exc}")
        return False


if __name__ == "__main__":
    trace = run_agent("I want a refund for order #4521, it never arrived.")
    print(trace.model_dump_json(indent=2))
    print(f"\n✅ Agent run complete. Status: {trace.status}, steps: {len(trace.steps)}, total_latency_ms: {trace.total_latency_ms}")
    send_trace_to_api(trace)