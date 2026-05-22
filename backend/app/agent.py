from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .config import get_settings
from .repository import get_repository

SESSION_MEMORY: dict[str, list[dict[str, Any]]] = {}

SYSTEM_PROMPT = """You are a friendly customer support assistant for an e-commerce store.

You have four tools:
- lookup_order: check the status of an order by its ID
- request_refund: process a refund for a delivered order (always look up the order first to confirm delivery)
- search_knowledge_base: find answers in product guides and store policies
- create_ticket: escalate to a human agent when you cannot resolve the issue

Guidelines:
- For refund requests, always call lookup_order first, then request_refund if delivered.
- If the knowledge base returns a low score or no result, create a ticket instead of guessing.
- If you already know the order ID from earlier in the conversation, use it directly.
- Be concise and helpful."""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "lookup_order",
        "description": "Look up an order by its ID to get status, shipping date, and delivery estimate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID (e.g. ORD-1001 or a 32-character hex ID)",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "request_refund",
        "description": "Request a refund for a delivered order. Validates delivery status before approving.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID"},
                "reason": {"type": "string", "description": "The reason for the refund"},
            },
            "required": ["order_id", "reason"],
        },
    },
    {
        "name": "search_knowledge_base",
        "description": "Search product guides and store policies for answers to customer questions.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The customer's question"}},
            "required": ["query"],
        },
    },
    {
        "name": "create_ticket",
        "description": "Create a support ticket for a human agent when the bot cannot resolve the issue.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Short description of the issue"},
                "description": {"type": "string", "description": "Full details of the problem"},
                "order_id": {
                    "type": "string",
                    "description": "Associated order ID if applicable",
                },
            },
            "required": ["subject", "description"],
        },
    },
]


@dataclass
class ToolEvent:
    name: str
    input: dict[str, Any]
    output: dict[str, Any]


def lookup_order(order_id: str) -> dict[str, Any]:
    order = get_repository().get_order(order_id)
    if not order:
        return {"found": False, "message": f"No order found for {order_id}."}
    return {"found": True, "order": order}


def request_refund(order_id: str, reason: str) -> dict[str, Any]:
    order = get_repository().get_order(order_id)
    if not order:
        return {"approved": False, "message": f"No order found for {order_id}."}
    if not order["delivered"]:
        return {
            "approved": False,
            "message": "Refunds can only start after delivery in the current release.",
        }
    if not reason.strip():
        return {"approved": False, "message": "A refund reason is required."}
    return {
        "approved": True,
        "reference": f"RFD-{order_id[-4:]}",
        "message": f"Refund request created for {order_id}.",
    }


def create_ticket(subject: str, description: str, order_id: str | None = None) -> dict[str, Any]:
    suffix = order_id or "GEN"
    return {
        "ticket_id": f"TCK-{suffix}",
        "subject": subject,
        "message": "Support ticket created for human follow-up.",
    }


def search_knowledge_base(query: str) -> list[dict[str, Any]]:
    return get_repository().search_knowledge(query)


def _execute_tool(name: str, inputs: dict[str, Any]) -> dict[str, Any] | list[Any]:
    if name == "lookup_order":
        return lookup_order(**inputs)
    if name == "request_refund":
        return request_refund(**inputs)
    if name == "search_knowledge_base":
        return search_knowledge_base(**inputs)
    if name == "create_ticket":
        return create_ticket(**inputs)
    return {"error": f"Unknown tool: {name}"}


def handle_message(session_id: str, message: str) -> dict[str, Any]:
    settings = get_settings()

    if not settings.anthropic_api_key:
        return _handle_message_deterministic(session_id, message)

    import anthropic

    history = SESSION_MEMORY.setdefault(session_id, [])
    history.append({"role": "user", "content": message})

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    tool_events: list[ToolEvent] = []
    messages: list[dict[str, Any]] = list(history)

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason != "tool_use":
            reply = next(
                (block.text for block in assistant_content if hasattr(block, "text")),
                "I'm sorry, I couldn't process that request.",
            )
            break

        tool_result_blocks: list[dict[str, Any]] = []
        for block in assistant_content:
            if block.type == "tool_use":
                result = _execute_tool(block.name, dict(block.input))
                tool_events.append(ToolEvent(block.name, dict(block.input), result))
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )

        messages.append({"role": "user", "content": tool_result_blocks})

    SESSION_MEMORY[session_id] = messages

    return {
        "reply": reply,
        "tool_events": [
            {"name": e.name, "input": e.input, "output": e.output} for e in tool_events
        ],
        "session_id": session_id,
    }


# ---------------------------------------------------------------------------
# Deterministic fallback — used when no Anthropic API key is configured
# (keeps in-memory tests and local-only demos working without an API key)
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "have",
    "how",
    "i",
    "is",
    "it",
    "my",
    "of",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "why",
    "you",
}

_FALLBACK_MEMORY: dict[str, list[dict[str, str]]] = {}
_MAX_HISTORY = 20


def _remember_fallback(session_id: str, role: str, content: str) -> None:
    history = _FALLBACK_MEMORY.setdefault(session_id, [])
    history.append({"role": role, "content": content})
    _FALLBACK_MEMORY[session_id] = history[-_MAX_HISTORY:]


def _recall_last_order_id(session_id: str) -> str | None:
    history = _FALLBACK_MEMORY.get(session_id, [])
    for item in reversed(history):
        match = _extract_order_id(item["content"])
        if match:
            return match
    return None


def _extract_order_id(text: str) -> str | None:
    upper = re.search(r"\bORD-\d{4}\b", text.upper())
    if upper:
        return upper.group(0)
    olist = re.search(r"\b[a-f0-9]{32}\b", text.lower())
    if olist:
        return olist.group(0)
    return None


def _handle_message_deterministic(session_id: str, message: str) -> dict[str, Any]:
    _remember_fallback(session_id, "user", message)
    tool_events: list[ToolEvent] = []
    order_id = _extract_order_id(message) or _recall_last_order_id(session_id)
    lower = message.lower()

    if any(p in lower for p in ["where is my order", "order status", "track my order"]):
        if not order_id:
            reply = "Please share your order ID so I can look it up."
        else:
            result = lookup_order(order_id)
            tool_events.append(ToolEvent("lookup_order", {"order_id": order_id}, result))
            if result["found"]:
                o = result["order"]
                reply = (
                    f"Order {order_id} is currently {o['status']}. "
                    f"It shipped on {o['shipping_date']} and is expected by {o['delivery_estimate']}."
                )
            else:
                reply = result["message"]
    elif "refund" in lower:
        if not order_id:
            reply = "I can help with that refund request. Please share the order ID first."
        else:
            result = request_refund(order_id, message)
            tool_events.append(
                ToolEvent("request_refund", {"order_id": order_id, "reason": message}, result)
            )
            reply = result["message"]
    elif any(p in lower for p in ["ticket", "human", "agent", "escalate"]):
        result = create_ticket("Customer support follow-up", message, order_id)
        tool_events.append(
            ToolEvent(
                "create_ticket",
                {"subject": "Customer support follow-up", "order_id": order_id},
                result,
            )
        )
        reply = f"{result['message']} Reference: {result['ticket_id']}."
    else:
        results = search_knowledge_base(message)
        tool_events.append(
            ToolEvent("search_knowledge_base", {"query": message}, {"matches": results})
        )
        if results and results[0]["score"] >= 0.25:
            top = results[0]
            reply = f"Here's the best answer I found from {top['title']}: {top['content']}"
        else:
            result = create_ticket("Low-confidence support query", message, order_id)
            tool_events.append(
                ToolEvent(
                    "create_ticket",
                    {"subject": "Low-confidence support query", "order_id": order_id},
                    result,
                )
            )
            reply = (
                "I'm not confident enough to answer that from the current knowledge base. "
                f"I created a follow-up ticket: {result['ticket_id']}."
            )

    _remember_fallback(session_id, "assistant", reply)
    return {
        "reply": reply,
        "tool_events": [
            {"name": e.name, "input": e.input, "output": e.output} for e in tool_events
        ],
        "session_id": session_id,
    }
