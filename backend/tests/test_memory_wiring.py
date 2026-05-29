"""Phase 8 — Memory wiring (TDD: red phase).

8a: ConversationStore replaces SESSION_MEMORY in the live agent.
8b: CustomerStore is wired in; customer context appears in the system prompt.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from backend.app.agent import SESSION_MEMORY, handle_message
from backend.app.conversation_store import InMemoryConversationStore
from backend.app.customer_store import InMemoryCustomerStore
from backend.app.data import KNOWLEDGE_BASE, ORDERS
from backend.app.repository import InMemoryRepository


def _in_memory_repo() -> InMemoryRepository:
    return InMemoryRepository(orders=ORDERS, knowledge_documents=KNOWLEDGE_BASE)


def _text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_use_block(tool_id: str, name: str, inputs: dict[str, Any]) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = inputs
    return block


def _response(stop_reason: str, *blocks: MagicMock) -> MagicMock:
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = list(blocks)
    return resp


class _AgentTestBase(unittest.TestCase):
    def setUp(self) -> None:
        SESSION_MEMORY.clear()
        self._repo = _in_memory_repo()
        self._repo_patcher = patch("backend.app.agent.get_repository", return_value=self._repo)
        self._repo_patcher.start()
        self._settings_patcher = patch(
            "backend.app.agent.get_settings",
            return_value=MagicMock(anthropic_api_key="test-key"),
        )
        self._settings_patcher.start()

    def tearDown(self) -> None:
        self._repo_patcher.stop()
        self._settings_patcher.stop()

    def _patch_anthropic(self, side_effects: list[MagicMock]) -> MagicMock:
        client = MagicMock()
        client.messages.create.side_effect = side_effects
        patcher = patch("anthropic.Anthropic", return_value=client)
        patcher.start()
        self.addCleanup(patcher.stop)
        return client


# ---------------------------------------------------------------------------
# 8a: ConversationStore wiring
# ---------------------------------------------------------------------------


class ConversationStoreWiringTests(_AgentTestBase):
    """handle_message must use an injected ConversationStore instead of SESSION_MEMORY."""

    def test_user_turn_saved_to_conv_store(self) -> None:
        conv_store = InMemoryConversationStore()
        self._patch_anthropic([_response("end_turn", _text_block("Hello there!"))])

        handle_message("sess-1", "Hi", conv_store=conv_store)

        turns = conv_store.load_turns("sess-1")
        user_turns = [t for t in turns if t["role"] == "user"]
        self.assertEqual(len(user_turns), 1)
        self.assertEqual(user_turns[0]["content"], "Hi")

    def test_assistant_reply_saved_to_conv_store(self) -> None:
        conv_store = InMemoryConversationStore()
        self._patch_anthropic([_response("end_turn", _text_block("I can help with that."))])

        handle_message("sess-1", "I need help", conv_store=conv_store)

        turns = conv_store.load_turns("sess-1")
        assistant_turns = [t for t in turns if t["role"] == "assistant"]
        self.assertEqual(len(assistant_turns), 1)
        self.assertEqual(assistant_turns[0]["content"], "I can help with that.")

    def test_second_call_loads_history_from_conv_store(self) -> None:
        conv_store = InMemoryConversationStore()
        client = self._patch_anthropic(
            [
                _response("end_turn", _text_block("ORD-1001 is delivered.")),
                _response("end_turn", _text_block("Your refund has been initiated.")),
            ]
        )

        handle_message("sess-1", "Status of ORD-1001?", conv_store=conv_store)
        handle_message("sess-1", "Can I get a refund?", conv_store=conv_store)

        # Second call's messages must include the first exchange
        second_call_messages = client.messages.create.call_args_list[1][1]["messages"]
        all_content = " ".join(
            m["content"] if isinstance(m["content"], str) else str(m["content"])
            for m in second_call_messages
        )
        self.assertIn("ORD-1001", all_content)

    def test_session_memory_not_populated_when_conv_store_injected(self) -> None:
        conv_store = InMemoryConversationStore()
        self._patch_anthropic([_response("end_turn", _text_block("Got it."))])

        handle_message("sess-isolated", "Hello", conv_store=conv_store)

        self.assertNotIn("sess-isolated", SESSION_MEMORY)


# ---------------------------------------------------------------------------
# 8b: CustomerStore wiring
# ---------------------------------------------------------------------------


class CustomerStoreWiringTests(_AgentTestBase):
    """handle_message must upsert the customer and inject context into the system prompt."""

    def test_customer_created_and_session_linked(self) -> None:
        customer_store = InMemoryCustomerStore()
        self._patch_anthropic([_response("end_turn", _text_block("Hi!"))])

        handle_message(
            "sess-alice",
            "Hello",
            customer_email="alice@example.com",
            customer_store=customer_store,
        )

        cid = customer_store.get_customer_id_for_session("sess-alice")
        self.assertIsNotNone(cid)
        customer = customer_store.get_customer(cid)
        self.assertIsNotNone(customer)
        self.assertEqual(customer["email"], "alice@example.com")

    def test_prior_orders_appear_in_system_prompt(self) -> None:
        customer_store = InMemoryCustomerStore()
        customer = customer_store.upsert_customer("alice@example.com", "Alice")
        customer_store.link_order(customer["customer_id"], "ORD-9999")

        client = self._patch_anthropic([_response("end_turn", _text_block("Hello!"))])

        handle_message(
            "sess-alice",
            "Hi",
            customer_email="alice@example.com",
            customer_store=customer_store,
        )

        system_prompt = client.messages.create.call_args[1]["system"]
        self.assertIn("ORD-9999", system_prompt)

    def test_memory_facts_appear_in_system_prompt(self) -> None:
        customer_store = InMemoryCustomerStore()
        customer = customer_store.upsert_customer("alice@example.com", "Alice")
        customer_store.save_memory_fact(
            customer_id=customer["customer_id"],
            fact_type="order_preference",
            fact_text="Prefers express shipping",
            confidence=0.9,
            source_session_id="prev-sess",
        )

        client = self._patch_anthropic([_response("end_turn", _text_block("Hello!"))])

        handle_message(
            "sess-alice",
            "Hi",
            customer_email="alice@example.com",
            customer_store=customer_store,
        )

        system_prompt = client.messages.create.call_args[1]["system"]
        self.assertIn("Prefers express shipping", system_prompt)

    def test_anonymous_session_still_works(self) -> None:
        self._patch_anthropic([_response("end_turn", _text_block("How can I help?"))])

        result = handle_message("anon-sess", "Hello")

        self.assertEqual(result["reply"], "How can I help?")

    def test_lookup_order_links_order_to_customer(self) -> None:
        customer_store = InMemoryCustomerStore()
        conv_store = InMemoryConversationStore()
        self._patch_anthropic(
            [
                _response(
                    "tool_use",
                    _tool_use_block("tu-1", "lookup_order", {"order_id": "ORD-1001"}),
                ),
                _response("end_turn", _text_block("Your order is on its way.")),
            ]
        )

        handle_message(
            "sess-alice",
            "Where is ORD-1001?",
            customer_email="alice@example.com",
            customer_store=customer_store,
            conv_store=conv_store,
        )

        cid = customer_store.get_customer_id_for_session("sess-alice")
        linked_orders = customer_store.get_customer_orders(cid)
        self.assertIn("ORD-1001", linked_orders)

    def test_returning_customer_no_double_upsert(self) -> None:
        customer_store = InMemoryCustomerStore()
        customer_store.upsert_customer("alice@example.com", "Alice")

        self._patch_anthropic(
            [
                _response("end_turn", _text_block("Welcome back!")),
                _response("end_turn", _text_block("Of course!")),
            ]
        )

        handle_message(
            "sess-1", "Hi", customer_email="alice@example.com", customer_store=customer_store
        )
        handle_message(
            "sess-2", "Help", customer_email="alice@example.com", customer_store=customer_store
        )

        # Both sessions should map to the same customer_id
        cid1 = customer_store.get_customer_id_for_session("sess-1")
        cid2 = customer_store.get_customer_id_for_session("sess-2")
        self.assertEqual(cid1, cid2)


if __name__ == "__main__":
    unittest.main()
