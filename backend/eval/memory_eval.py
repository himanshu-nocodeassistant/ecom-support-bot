"""Phase 8g — Memory recall evaluation.

memory_recall_rate: for each fixture, pre-populate CustomerStore with stored_facts
and prior_orders, then call handle_message and verify that every expected_context_fragment
appears in the system prompt passed to Claude.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

from backend.app.agent import handle_message
from backend.app.conversation_store import InMemoryConversationStore
from backend.app.customer_store import InMemoryCustomerStore


def memory_recall_rate(fixtures_recalled: int, fixtures_total: int) -> float:
    if fixtures_total == 0:
        return 0.0
    return fixtures_recalled / fixtures_total


def _check_fixture(fixture: dict[str, Any]) -> bool:
    """Return True if all expected_context_fragments appear in the system prompt."""
    expected: list[str] = fixture.get("expected_context_fragments", [])
    stored_facts: list[dict] = fixture.get("stored_facts", [])
    prior_orders: list[str] = fixture.get("prior_orders", [])

    customer_store = InMemoryCustomerStore()
    conv_store = InMemoryConversationStore()
    email = f"eval-{uuid.uuid4().hex[:8]}@example.com"

    customer = customer_store.upsert_customer(email, "Eval User")
    cid = customer["customer_id"]

    for fact in stored_facts:
        customer_store.save_memory_fact(
            customer_id=cid,
            fact_type=fact["fact_type"],
            fact_text=fact["fact_text"],
            confidence=fact["confidence"],
            source_session_id="fixture-seed",
        )

    for order_id in prior_orders:
        customer_store.link_order(cid, order_id)

    # Capture the system prompt passed to Claude
    captured_system: list[str] = []

    def _capture(*args, **kwargs):
        captured_system.append(kwargs.get("system", ""))
        resp = MagicMock()
        resp.stop_reason = "end_turn"
        block = MagicMock()
        block.type = "text"
        block.text = "OK"
        resp.content = [block]
        return resp

    with patch("anthropic.Anthropic") as MockClient:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = _capture
        MockClient.return_value = mock_client

        with patch(
            "backend.app.agent.get_settings",
            return_value=MagicMock(anthropic_api_key="eval-key"),
        ):
            handle_message(
                session_id=f"eval-{uuid.uuid4().hex[:8]}",
                message="Hello",
                customer_email=email,
                customer_store=customer_store,
                conv_store=conv_store,
            )

    if not captured_system:
        return len(expected) == 0

    system_prompt = captured_system[0]

    if not expected:
        return True

    return all(fragment in system_prompt for fragment in expected)


def run_memory_eval(fixtures: list[dict[str, Any]]) -> dict[str, Any]:
    """Run all memory fixtures and return recall metrics."""
    total = len(fixtures)
    recalled = 0

    for fixture in fixtures:
        if _check_fixture(fixture):
            recalled += 1

    rate = memory_recall_rate(fixtures_recalled=recalled, fixtures_total=total)
    return {
        "memory_recall_rate": rate,
        "recalled": recalled,
        "total": total,
    }
