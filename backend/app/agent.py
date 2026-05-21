from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .data import KNOWLEDGE_BASE, ORDERS


SESSION_MEMORY: dict[str, list[dict[str, str]]] = {}
MAX_HISTORY = 20


@dataclass
class ToolEvent:
    name: str
    input: dict[str, Any]
    output: dict[str, Any]


def remember(session_id: str, role: str, content: str) -> None:
    history = SESSION_MEMORY.setdefault(session_id, [])
    history.append({"role": role, "content": content})
    SESSION_MEMORY[session_id] = history[-MAX_HISTORY:]


def recall_last_order_id(session_id: str) -> str | None:
    history = SESSION_MEMORY.get(session_id, [])
    for item in reversed(history):
        match = extract_order_id(item["content"])
        if match:
            return match
    return None


def extract_order_id(text: str) -> str | None:
    match = re.search(r"\bORD-\d{4}\b", text.upper())
    return match.group(0) if match else None


def lookup_order(order_id: str) -> dict[str, Any]:
    order = ORDERS.get(order_id)
    if not order:
        return {"found": False, "message": f"No order found for {order_id}."}
    return {"found": True, "order": order}


def request_refund(order_id: str, reason: str) -> dict[str, Any]:
    order = ORDERS.get(order_id)
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
    query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
    ranked = []
    for doc in KNOWLEDGE_BASE:
        haystack = f"{doc['title']} {doc['content']}".lower()
        score = sum(1 for word in query_words if word in haystack)
        if score:
            ranked.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "category": doc["category"],
                    "score": score,
                    "content": doc["content"],
                }
            )
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:3]


def handle_message(session_id: str, message: str) -> dict[str, Any]:
    remember(session_id, "user", message)
    tool_events: list[ToolEvent] = []
    order_id = extract_order_id(message) or recall_last_order_id(session_id)
    lower = message.lower()

    if any(phrase in lower for phrase in ["where is my order", "order status", "track my order"]):
        if not order_id:
            reply = "Please share your order ID in the format ORD-1001 so I can look it up."
        else:
            result = lookup_order(order_id)
            tool_events.append(ToolEvent("lookup_order", {"order_id": order_id}, result))
            if result["found"]:
                order = result["order"]
                reply = (
                    f"Order {order_id} is currently {order['status']}. "
                    f"It shipped on {order['shipping_date']} and is expected by {order['delivery_estimate']}."
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
    elif any(phrase in lower for phrase in ["ticket", "human", "agent", "escalate"]):
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
        if results and results[0]["score"] >= 2:
            top = results[0]
            reply = (
                f"Here’s the best answer I found from {top['title']}: {top['content']}"
            )
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
                "I’m not confident enough to answer that from the current knowledge base. "
                f"I created a follow-up ticket: {result['ticket_id']}."
            )

    remember(session_id, "assistant", reply)
    return {
        "reply": reply,
        "tool_events": [
            {"name": event.name, "input": event.input, "output": event.output}
            for event in tool_events
        ],
        "session_id": session_id,
    }
