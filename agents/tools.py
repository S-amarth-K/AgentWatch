"""
Mock tool implementations for the support agent demo.

In a real deployment these would hit a database / order-management API.
Here they return realistic-looking fake data so we can generate traces,
including deliberately breakable scenarios (order not found, refund
already issued, etc.) to produce failure-mode examples later.
"""

from __future__ import annotations

import random
import time

# A small fake "database" of orders, seeded so results are reproducible.
_FAKE_ORDERS = {
    "4521": {"status": "shipped", "delivery_estimate": "2026-07-10", "amount": 49.99, "refunded": False},
    "1001": {"status": "delivered", "delivery_estimate": "2026-07-15", "amount": 19.99, "refunded": False},
    "2002": {"status": "processing", "delivery_estimate": "2026-07-25", "amount": 89.50, "refunded": False},
    "3003": {"status": "shipped", "delivery_estimate": "2026-07-05", "amount": 120.00, "refunded": True},
}


def check_order_status(order_id: str) -> dict:
    """Look up the status of an order by its ID."""
    time.sleep(random.uniform(0.05, 0.15))  # simulate network latency
    order = _FAKE_ORDERS.get(order_id)
    if order is None:
        return {"error": f"No order found with id {order_id}"}
    return {
        "order_id": order_id,
        "status": order["status"],
        "delivery_estimate": order["delivery_estimate"],
        "amount": order["amount"],
        "already_refunded": order["refunded"],
    }


def issue_refund(order_id: str, reason: str) -> dict:
    """Issue a refund for a given order."""
    time.sleep(random.uniform(0.05, 0.15))
    order = _FAKE_ORDERS.get(order_id)
    if order is None:
        return {"error": f"No order found with id {order_id}"}
    if order["refunded"]:
        return {"error": f"Order {order_id} has already been refunded. Cannot refund twice."}
    order["refunded"] = True
    return {"refund_status": "success", "order_id": order_id, "amount": order["amount"], "reason": reason}


def escalate_to_human(order_id: str, issue_summary: str) -> dict:
    """Escalate a case to a human support agent."""
    time.sleep(random.uniform(0.05, 0.15))
    return {"escalation_status": "created", "order_id": order_id, "summary": issue_summary, "ticket_id": f"TICKET-{random.randint(1000, 9999)}"}


# Registry mapping tool name -> callable, used by the agent loop to dispatch.
TOOL_REGISTRY = {
    "check_order_status": check_order_status,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}

# Gemini function-declaration schemas (used to tell the model what tools exist).
TOOL_DECLARATIONS = [
    {
        "name": "check_order_status",
        "description": "Look up the current status, delivery estimate, and refund status of an order by its order ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID to look up."}
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Issue a refund for a given order. Only call this after verifying the order status supports a refund.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID to refund."},
                "reason": {"type": "string", "description": "Reason for the refund, e.g. 'not_delivered'."},
            },
            "required": ["order_id", "reason"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate the case to a human support agent when the issue cannot be resolved automatically.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID related to the issue."},
                "issue_summary": {"type": "string", "description": "Brief summary of the unresolved issue."},
            },
            "required": ["order_id", "issue_summary"],
        },
    },
]