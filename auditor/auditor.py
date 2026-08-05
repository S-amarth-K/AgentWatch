"""
LLM-based auditor: reads a full trace and classifies WHY it looks wrong,
producing a structured failure_mode + human-readable explanation.

This is deliberately a separate model call from the agent itself -- the
auditor is reviewing the agent's work after the fact, with the benefit of
seeing the whole trace at once (all steps, all tool results, the final
response), rather than reasoning step-by-step like the agent did.

MOCK_MODE:
    True  -> heuristic rule-based classification, no API calls. Useful for
             testing the pipeline and for traces where the failure is
             mechanically obvious (e.g. a tool literally returned an error).
    False -> real Gemini 2.5 Flash call, given the full trace and asked to
             return structured JSON matching the taxonomy.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from auditor.taxonomy import FAILURE_MODE_DESCRIPTIONS, format_taxonomy_for_prompt  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MOCK_MODE = False  # flip to False to use real Gemini classification


AUDITOR_SYSTEM_PROMPT = f"""You are an AI agent auditor. You will be given the
full execution trace of an autonomous support agent -- its reasoning at each
step, which tools it called, what those tools returned, and its final
response to the user.

Your job is to determine whether the agent's behavior falls into one of the
following failure categories, or whether it behaved correctly (\"none\"):

{format_taxonomy_for_prompt()}

Respond with ONLY a JSON object in this exact format, with no other text:
{{"failure_mode": "<one of the category names above>", "explanation": "<one or two sentences explaining your reasoning, referencing specific steps/tool results>"}}
"""


def _format_trace_for_prompt(trace: dict) -> str:
    """Renders a trace's steps and outcome as readable text for the auditor prompt."""
    lines = [f"User request: {trace.get('user_input')}\n"]
    for step in trace.get("steps", []):
        lines.append(f"Step {step['step_number']}:")
        lines.append(f"  Reasoning: {step.get('decision_rationale')}")
        if step.get("tool_called"):
            lines.append(f"  Tool called: {step['tool_called']}({step.get('tool_args')})")
            lines.append(f"  Tool result: {step.get('tool_result')}")
        lines.append(f"  Step status: {step.get('step_status')}")
    lines.append(f"\nFinal response to user: {trace.get('final_response')}")
    lines.append(f"Overall trace status: {trace.get('status')}")
    return "\n".join(lines)


def classify_trace_mock(trace: dict) -> dict:
    """
    Heuristic rule-based classification -- no API calls. Good enough to
    catch mechanically obvious failures (loops, tool-level errors) for
    testing the pipeline without burning API quota.
    """
    steps = trace.get("steps", [])

    # Infrastructure-level failures (API rate limits, timeouts, network
    # errors) must be checked FIRST -- these mean the agent's reasoning loop
    # never completed properly, so they're not a reasoning/tool-usage
    # failure at all, and shouldn't be misclassified as "none" just because
    # no tool_result happens to contain an "error" key.
    if trace.get("status") == "error" and trace.get("error_message"):
        return {
            "failure_mode": "infrastructure_error",
            "explanation": f"The agent's execution was interrupted by an infrastructure-level failure, not a reasoning error: {trace.get('error_message')[:200]}",
        }

    # Loop detection: same tool called consecutively more than once.
    tool_sequence = [s.get("tool_called") for s in steps if s.get("tool_called")]
    for i in range(1, len(tool_sequence)):
        if tool_sequence[i] == tool_sequence[i - 1]:
            return {
                "failure_mode": "infinite_loop",
                "explanation": f"The agent called '{tool_sequence[i]}' repeatedly in consecutive steps without new information, suggesting it was stuck rather than making progress.",
            }

    # Tool-level errors -> distinguish policy violation vs wrong tool call.
    for step in steps:
        result = step.get("tool_result")
        if isinstance(result, dict) and "error" in result:
            error_text = str(result["error"]).lower()
            if "already been refunded" in error_text:
                return {
                    "failure_mode": "policy_violation",
                    "explanation": f"The agent attempted to refund an order that had already been refunded, violating the no-double-refund policy. Tool result: {result['error']}",
                }
            if "no order found" in error_text:
                return {
                    "failure_mode": "wrong_tool_call",
                    "explanation": f"The agent called a tool with an order ID that does not exist, resulting in: {result['error']}",
                }

    if trace.get("status") == "incomplete":
        return {
            "failure_mode": "low_confidence_answered_anyway",
            "explanation": "The agent did not reach a final response within the allowed steps, suggesting it was uncertain how to proceed.",
        }

    return {"failure_mode": "none", "explanation": "No issues detected -- tool usage and final response are consistent with the data available."}


def classify_trace_real(trace: dict) -> dict:
    """Real Gemini 2.5 Flash call, asked to return structured JSON."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"failure_mode": None, "explanation": "GEMINI_API_KEY not set."}

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=30_000))

    trace_text = _format_trace_for_prompt(trace)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=trace_text)])],
        config=types.GenerateContentConfig(
            system_instruction=AUDITOR_SYSTEM_PROMPT,
            response_mime_type="application/json",
        ),
    )

    try:
        parsed = json.loads(response.text)
        if parsed.get("failure_mode") not in FAILURE_MODE_DESCRIPTIONS:
            parsed["failure_mode"] = "none"
        return parsed
    except (json.JSONDecodeError, AttributeError) as exc:
        return {"failure_mode": None, "explanation": f"Failed to parse auditor response: {exc}"}


def classify_trace(trace: dict) -> dict:
    """Entry point: dispatches to mock or real classification based on MOCK_MODE."""
    if MOCK_MODE:
        return classify_trace_mock(trace)
    return classify_trace_real(trace)


if __name__ == "__main__":
    # Quick self-test with a hand-built example of a policy-violation trace.
    example_trace = {
        "user_input": "Refund order #3003, it's broken.",
        "steps": [
            {
                "step_number": 1,
                "decision_rationale": "Checking order status before refunding.",
                "tool_called": "check_order_status",
                "tool_args": {"order_id": "3003"},
                "tool_result": {"status": "shipped", "already_refunded": True},
                "step_status": "success",
            },
            {
                "step_number": 2,
                "decision_rationale": "Attempting refund.",
                "tool_called": "issue_refund",
                "tool_args": {"order_id": "3003", "reason": "damaged"},
                "tool_result": {"error": "Order 3003 has already been refunded."},
                "step_status": "failure",
            },
        ],
        "final_response": "Something went wrong: Order 3003 has already been refunded.",
        "status": "error",
    }
    result = classify_trace(example_trace)
    print(json.dumps(result, indent=2))
    print("\n✅ Auditor classification works.")