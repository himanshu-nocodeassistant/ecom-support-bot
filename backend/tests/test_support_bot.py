import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from backend.app.agent import SESSION_MEMORY, handle_message
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


def _make_anthropic_client(side_effects: list[MagicMock]) -> MagicMock:
    client = MagicMock()
    client.messages.create.side_effect = side_effects
    return client


class SupportBotTests(unittest.TestCase):
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

    def _patch_anthropic(self, side_effects: list[MagicMock]) -> None:
        client = _make_anthropic_client(side_effects)
        patcher = patch("anthropic.Anthropic", return_value=client)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_product_question_uses_kb(self) -> None:
        kb_results = self._repo.search_knowledge("Does the portable blender have a safety lock?")
        self._patch_anthropic(
            [
                _response(
                    "tool_use",
                    _tool_use_block(
                        "tu-1",
                        "search_knowledge_base",
                        {"query": "Does the portable blender have a safety lock?"},
                    ),
                ),
                _response(
                    "end_turn",
                    _text_block(
                        f"Here's the best answer I found from {kb_results[0]['title']}: {kb_results[0]['content']}"
                    ),
                ),
            ]
        )

        result = handle_message("session-product", "Does the portable blender have a safety lock?")

        self.assertIn("portable blender", result["reply"].lower())
        self.assertEqual(result["tool_events"][0]["name"], "search_knowledge_base")

    def test_order_status_lookup(self) -> None:
        self._patch_anthropic(
            [
                _response(
                    "tool_use",
                    _tool_use_block("tu-1", "lookup_order", {"order_id": "ORD-1001"}),
                ),
                _response("end_turn", _text_block("Order ORD-1001 is currently in_transit.")),
            ]
        )

        result = handle_message("session-order", "Where is my order ORD-1001?")

        self.assertIn("ORD-1001", result["reply"])
        self.assertEqual(result["tool_events"][0]["name"], "lookup_order")

    def test_multi_step_refund_flow(self) -> None:
        """Claude chains lookup_order then request_refund in a single response turn."""
        self._patch_anthropic(
            [
                _response(
                    "tool_use",
                    _tool_use_block("tu-1", "lookup_order", {"order_id": "ORD-1002"}),
                    _tool_use_block(
                        "tu-2",
                        "request_refund",
                        {"order_id": "ORD-1002", "reason": "item arrived damaged"},
                    ),
                ),
                _response(
                    "end_turn",
                    _text_block("Refund request created for ORD-1002. Reference: RFD-1002."),
                ),
            ]
        )

        result = handle_message(
            "session-refund", "My order ORD-1002 arrived damaged, I want a refund."
        )

        names = [e["name"] for e in result["tool_events"]]
        self.assertIn("lookup_order", names)
        self.assertIn("request_refund", names)
        self.assertIn("Refund request created", result["reply"])

    def test_memory_recalls_order_id_for_follow_up(self) -> None:
        """Second turn: Claude uses order ID from session history without user repeating it."""
        self._patch_anthropic(
            [
                # Turn 1: order lookup
                _response(
                    "tool_use",
                    _tool_use_block("tu-1", "lookup_order", {"order_id": "ORD-1002"}),
                ),
                _response("end_turn", _text_block("Order ORD-1002 has been delivered.")),
                # Turn 2: refund (Claude reads order ID from history)
                _response(
                    "tool_use",
                    _tool_use_block(
                        "tu-2",
                        "request_refund",
                        {"order_id": "ORD-1002", "reason": "Can I get a refund for it?"},
                    ),
                ),
                _response(
                    "end_turn",
                    _text_block("Refund request created for ORD-1002."),
                ),
            ]
        )

        handle_message("session-memory", "Where is my order ORD-1002?")
        result = handle_message("session-memory", "Can I get a refund for it?")

        self.assertEqual(result["tool_events"][0]["name"], "request_refund")
        self.assertIn("Refund request created", result["reply"])

    def test_low_confidence_creates_ticket(self) -> None:
        self._patch_anthropic(
            [
                _response(
                    "tool_use",
                    _tool_use_block(
                        "tu-1",
                        "search_knowledge_base",
                        {"query": "Can you explain your wholesale partner rebate schedule?"},
                    ),
                ),
                _response(
                    "tool_use",
                    _tool_use_block(
                        "tu-2",
                        "create_ticket",
                        {
                            "subject": "Wholesale rebate schedule inquiry",
                            "description": "Can you explain your wholesale partner rebate schedule?",
                        },
                    ),
                ),
                _response(
                    "end_turn",
                    _text_block(
                        "I'm not confident enough to answer that. I created a follow-up ticket: TCK-GEN."
                    ),
                ),
            ]
        )

        result = handle_message(
            "session-low-confidence", "Can you explain your wholesale partner rebate schedule?"
        )

        names = [e["name"] for e in result["tool_events"]]
        self.assertIn("create_ticket", names)
        self.assertIn("not confident enough", result["reply"].lower())


if __name__ == "__main__":
    unittest.main()
