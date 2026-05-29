from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from .config import get_settings
from .conversation_store import ConversationStore, InMemoryConversationStore
from .customer_store import CustomerStore, InMemoryCustomerStore
from .data import KNOWLEDGE_BASE, ORDERS
from .memory_context import build_customer_context, build_system_prompt
from .prompts import SYSTEM_PROMPT
from .repository import InMemoryRepository, PostgresRepository, get_repository

SESSION_MEMORY: dict[str, list[dict[str, Any]]] = {}

_default_conv_store: ConversationStore = InMemoryConversationStore()
_default_customer_store: CustomerStore = InMemoryCustomerStore()

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


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


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


def search_knowledge_base(query: str, repo=None) -> list[dict[str, Any]]:
    r = repo if repo is not None else get_repository()
    return r.search_knowledge(query)


def _execute_tool(name: str, inputs: dict[str, Any], repo=None) -> dict[str, Any] | list[Any]:
    if name == "lookup_order":
        return lookup_order(**inputs)
    if name == "request_refund":
        return request_refund(**inputs)
    if name == "search_knowledge_base":
        return search_knowledge_base(repo=repo, **inputs)
    if name == "create_ticket":
        return create_ticket(**inputs)
    return {"error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# Repository selection by mode
# ---------------------------------------------------------------------------


def _repo_for_mode(mode: str):
    """Return the repository instance appropriate for the requested mode."""
    settings = get_settings()
    fallback = InMemoryRepository(orders=ORDERS, knowledge_documents=KNOWLEDGE_BASE)

    if mode == "phase1":
        return fallback

    if mode in ("phase2", "phase3", "phase4"):
        if settings.data_backend == "postgres" and settings.database_url:
            return PostgresRepository(
                database_url=settings.database_url,
                fallback=fallback,
                voyage_api_key=settings.voyage_api_key
                if mode != "phase2" or settings.voyage_api_key
                else None,
            )
        return fallback

    return get_repository()


# ---------------------------------------------------------------------------
# Phase 3 / 4 — Claude agent tool loop (synchronous)
# ---------------------------------------------------------------------------


def handle_message(
    session_id: str,
    message: str,
    mode: str = "phase3",
    customer_email: str | None = None,
    conv_store: ConversationStore | None = None,
    customer_store: CustomerStore | None = None,
) -> dict[str, Any]:
    settings = get_settings()

    if mode in ("phase1", "phase2") or not settings.anthropic_api_key:
        return _handle_message_deterministic(session_id, message, mode=mode)

    import anthropic

    repo = _repo_for_mode(mode)

    # --- conversation history ---
    if conv_store is not None:
        prior_turns = conv_store.load_turns(session_id)
        messages: list[dict[str, Any]] = list(prior_turns) + [{"role": "user", "content": message}]
    else:
        history = SESSION_MEMORY.setdefault(session_id, [])
        history.append({"role": "user", "content": message})
        messages = list(history)

    # --- customer context ---
    customer_id: str | None = None
    system_prompt_text = SYSTEM_PROMPT
    if customer_email and customer_store is not None:
        customer = customer_store.upsert_customer(email=customer_email, name="")
        customer_id = customer["customer_id"]
        customer_store.link_session(session_id, customer_id)
        facts = customer_store.load_memory_facts(customer_id)
        prior_orders = customer_store.get_customer_orders(customer_id)
        ctx = build_customer_context(facts=facts, prior_order_ids=prior_orders)
        system_prompt_text = build_system_prompt(customer_context=ctx)

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    tool_events: list[ToolEvent] = []
    reply = "I'm sorry, I couldn't process that request."

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system_prompt_text,
            tools=TOOLS,
            messages=messages,
        )

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason != "tool_use":
            reply = next(
                (block.text for block in assistant_content if hasattr(block, "text")),
                reply,
            )
            break

        tool_result_blocks: list[dict[str, Any]] = []
        for block in assistant_content:
            if block.type == "tool_use":
                result = _execute_tool(block.name, dict(block.input), repo=repo)
                tool_events.append(ToolEvent(block.name, dict(block.input), result))
                if (
                    block.name == "lookup_order"
                    and customer_id is not None
                    and customer_store is not None
                ):
                    order_id = dict(block.input).get("order_id")
                    if order_id:
                        customer_store.link_order(customer_id, order_id)
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )

        messages.append({"role": "user", "content": tool_result_blocks})

    # --- persist turns ---
    if conv_store is not None:
        conv_store.save_turn(session_id, "user", message)
        conv_store.save_turn(session_id, "assistant", reply)
    else:
        SESSION_MEMORY[session_id] = messages

    return {
        "reply": reply,
        "tool_events": [
            {"name": e.name, "input": e.input, "output": e.output} for e in tool_events
        ],
        "sources": _extract_sources(tool_events),
        "session_id": session_id,
        "mode": mode,
    }


# ---------------------------------------------------------------------------
# Phase 4 — SSE streaming generator
# ---------------------------------------------------------------------------


async def handle_message_stream(
    session_id: str,
    message: str,
    customer_email: str | None = None,
    conv_store: ConversationStore | None = None,
    customer_store: CustomerStore | None = None,
):
    """Async generator yielding SSE-formatted event strings for Phase 4."""
    settings = get_settings()

    if not settings.anthropic_api_key:
        result = _handle_message_deterministic(session_id, message, mode="phase3")
        yield _sse("token", {"text": result["reply"]})
        for e in result["tool_events"]:
            yield _sse(
                "tool_result", {"name": e["name"], "input": e["input"], "output": e["output"]}
            )
        yield _sse("done", {"session_id": session_id, "tool_count": len(result["tool_events"])})
        return

    import anthropic

    repo = _repo_for_mode("phase4")

    # --- conversation history ---
    if conv_store is not None:
        prior_turns = conv_store.load_turns(session_id)
        messages: list[dict[str, Any]] = list(prior_turns) + [{"role": "user", "content": message}]
    else:
        history = SESSION_MEMORY.setdefault(session_id, [])
        history.append({"role": "user", "content": message})
        messages = list(history)

    # --- customer context ---
    customer_id: str | None = None
    returning_customer = False
    system_prompt_text = SYSTEM_PROMPT
    if customer_email and customer_store is not None:
        customer = customer_store.upsert_customer(email=customer_email, name="")
        customer_id = customer["customer_id"]
        customer_store.link_session(session_id, customer_id)
        facts = customer_store.load_memory_facts(customer_id)
        prior_orders = customer_store.get_customer_orders(customer_id)
        returning_customer = bool(facts or prior_orders)
        ctx = build_customer_context(facts=facts, prior_order_ids=prior_orders)
        system_prompt_text = build_system_prompt(customer_context=ctx)

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    tool_events: list[ToolEvent] = []
    final_reply = ""

    try:
        while True:
            # Collect the full response for tool handling, stream tokens as they arrive
            tool_result_blocks: list[dict[str, Any]] = []
            assistant_content = []
            stop_reason = None

            with client.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=system_prompt_text,
                tools=TOOLS,
                messages=messages,
            ) as stream:
                current_tool_use: dict[str, Any] | None = None
                current_input_json = ""

                for event in stream:
                    etype = event.type

                    if etype == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            current_tool_use = {"id": block.id, "name": block.name}
                            current_input_json = ""
                            yield _sse("tool_start", {"name": block.name})
                        elif block.type == "text":
                            current_tool_use = None

                    elif etype == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            final_reply += delta.text
                            yield _sse("token", {"text": delta.text})
                        elif delta.type == "input_json_delta":
                            current_input_json += delta.partial_json

                    elif etype == "content_block_stop":
                        if current_tool_use is not None:
                            try:
                                parsed_input = (
                                    json.loads(current_input_json) if current_input_json else {}
                                )
                            except json.JSONDecodeError:
                                parsed_input = {}
                            current_tool_use["input"] = parsed_input
                            result = _execute_tool(
                                current_tool_use["name"], parsed_input, repo=repo
                            )
                            tool_events.append(
                                ToolEvent(current_tool_use["name"], parsed_input, result)
                            )
                            if (
                                current_tool_use["name"] == "lookup_order"
                                and customer_id is not None
                                and customer_store is not None
                            ):
                                order_id = parsed_input.get("order_id")
                                if order_id:
                                    customer_store.link_order(customer_id, order_id)
                            yield _sse(
                                "tool_result",
                                {
                                    "name": current_tool_use["name"],
                                    "input": parsed_input,
                                    "output": result,
                                },
                            )
                            tool_result_blocks.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": current_tool_use["id"],
                                    "content": json.dumps(result),
                                }
                            )
                            current_tool_use = None

                final_message = stream.get_final_message()
                assistant_content = final_message.content
                stop_reason = final_message.stop_reason

            messages.append({"role": "assistant", "content": assistant_content})

            if stop_reason != "tool_use":
                break

            messages.append({"role": "user", "content": tool_result_blocks})

        # --- persist turns ---
        if conv_store is not None:
            conv_store.save_turn(session_id, "user", message)
            conv_store.save_turn(session_id, "assistant", final_reply)
        else:
            SESSION_MEMORY[session_id] = messages

        yield _sse(
            "done",
            {
                "session_id": session_id,
                "tool_count": len(tool_events),
                "sources": _extract_sources(tool_events),
                "returning_customer": returning_customer,
            },
        )

    except Exception as exc:
        yield _sse("error", {"message": str(exc)})


def _extract_sources(tool_events: list[ToolEvent]) -> list[dict[str, Any]]:
    """Collect unique cited sources from search_knowledge_base tool results.

    Returns list of {title, category, score} dicts, sorted by score desc.
    Sources with score < 0.15 are suppressed (too weak to cite confidently).
    """
    seen: set[str] = set()
    sources: list[dict[str, Any]] = []
    for evt in tool_events:
        if evt.name != "search_knowledge_base":
            continue
        results = evt.output if isinstance(evt.output, list) else []
        for r in results:
            title = r.get("title", "")
            score = float(r.get("score", 0))
            if score < 0.15 or title in seen:
                continue
            seen.add(title)
            sources.append(
                {
                    "title": title,
                    "category": r.get("category", ""),
                    "score": round(score, 4),
                }
            )
    return sorted(sources, key=lambda s: s["score"], reverse=True)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Compare — run phase1, phase2, phase3 concurrently
# ---------------------------------------------------------------------------


async def handle_compare(message: str) -> dict[str, Any]:
    loop = asyncio.get_event_loop()

    def run(mode: str) -> dict[str, Any]:
        import uuid

        session_id = f"compare-{mode}-{uuid.uuid4().hex[:8]}"
        return handle_message(session_id, message, mode=mode)

    results = await asyncio.gather(
        loop.run_in_executor(None, run, "phase1"),
        loop.run_in_executor(None, run, "phase2"),
        loop.run_in_executor(None, run, "phase3"),
    )

    return {
        "phase1": results[0],
        "phase2": results[1],
        "phase3": results[2],
    }


# ---------------------------------------------------------------------------
# Deterministic fallback — phase1 and phase2, and no-API-key path
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


def _handle_message_deterministic(
    session_id: str, message: str, mode: str = "phase1"
) -> dict[str, Any]:
    _remember_fallback(session_id, "user", message)
    repo = _repo_for_mode(mode)
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
        results = search_knowledge_base(message, repo=repo)
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
        "sources": _extract_sources(tool_events),
        "session_id": session_id,
        "mode": mode,
    }
