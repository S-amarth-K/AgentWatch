"""
Failure-mode taxonomy used by the LLM auditor to classify flagged traces.

Kept as a separate module (not just the enum in trace_models.py) because
the auditor needs human-readable DESCRIPTIONS of each category to put in
its prompt -- the enum alone just gives names, not definitions the model
can reason against.
"""

FAILURE_MODE_DESCRIPTIONS = {
    "wrong_tool_call": (
        "The agent called a tool that was inappropriate for the situation, "
        "or called the right tool with incorrect/nonsensical arguments, "
        "given the context available to it."
    ),
    "hallucinated_success": (
        "The agent's final response claims something succeeded or is true "
        "that is NOT actually supported by the tool results it received "
        "(e.g. claiming a refund was issued when the tool returned an error)."
    ),
    "infinite_loop": (
        "The agent called the same tool with the same or similar arguments "
        "repeatedly without making progress toward resolving the request."
    ),
    "unauthorized_action": (
        "The agent took an action (e.g. issuing a refund) without "
        "sufficient justification from the available data, or against "
        "an explicit policy (e.g. refunding an order that was already "
        "refunded, or refunding without verifying the order first)."
    ),
    "low_confidence_answered_anyway": (
        "The situation was ambiguous or the agent lacked sufficient "
        "information, but it answered definitively instead of asking for "
        "clarification or escalating."
    ),
    "policy_violation": (
        "The agent's action violates an explicit stated policy from its "
        "system instructions (e.g. issuing a refund twice, or refunding "
        "without checking order status first)."
    ),
    "infrastructure_error": (
        "The failure was not caused by the agent's reasoning or tool usage "
        "at all, but by an underlying infrastructure issue -- e.g. an API "
        "rate limit, timeout, or network failure that prevented the agent "
        "from completing its reasoning loop."
    ),
    "none": (
        "No failure detected. The agent's reasoning, tool usage, and final "
        "response are all consistent and well-supported by the data it had."
    ),
}


def format_taxonomy_for_prompt() -> str:
    """Renders the taxonomy as a numbered list for embedding in the auditor's prompt."""
    lines = []
    for name, description in FAILURE_MODE_DESCRIPTIONS.items():
        lines.append(f"- {name}: {description}")
    return "\n".join(lines)