"""Phase 8c — Fact extractor (TDD: red phase).

extract_facts_from_conversation() takes a completed conversation (list of
{role, content} dicts) and returns a list of MemoryFact dicts that the agent
observed with enough confidence to persist.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


def _text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _response(*blocks: MagicMock) -> MagicMock:
    resp = MagicMock()
    resp.content = list(blocks)
    return resp


SAMPLE_CONVERSATION = [
    {"role": "user", "content": "Hi, I always prefer express shipping. My name is Alice."},
    {"role": "assistant", "content": "Got it, Alice! I'll keep that in mind."},
    {"role": "user", "content": "I had a damaged item last month, order ORD-1001."},
    {"role": "assistant", "content": "I'm sorry to hear that. I've created a ticket for you."},
]

EMPTY_CONVERSATION: list = []

SHORT_CONVERSATION = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]


class FactExtractorTests(unittest.TestCase):
    """extract_facts_from_conversation returns MemoryFact dicts."""

    def _patch_anthropic_with_json(self, json_text: str) -> None:
        client = MagicMock()
        client.messages.create.return_value = _response(_text_block(json_text))
        patcher = patch("anthropic.Anthropic", return_value=client)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_returns_list(self) -> None:
        from backend.app.fact_extractor import extract_facts_from_conversation

        self._patch_anthropic_with_json("[]")
        result = extract_facts_from_conversation(EMPTY_CONVERSATION, api_key="test-key")
        self.assertIsInstance(result, list)

    def test_empty_conversation_returns_empty_list(self) -> None:
        from backend.app.fact_extractor import extract_facts_from_conversation

        result = extract_facts_from_conversation(EMPTY_CONVERSATION, api_key="test-key")
        self.assertEqual(result, [])

    def test_short_conversation_returns_empty_list(self) -> None:
        from backend.app.fact_extractor import extract_facts_from_conversation

        result = extract_facts_from_conversation(SHORT_CONVERSATION, api_key="test-key")
        self.assertEqual(result, [])

    def test_extracts_facts_from_model_response(self) -> None:
        from backend.app.fact_extractor import extract_facts_from_conversation

        self._patch_anthropic_with_json(
            '[{"fact_type": "order_preference", "fact_text": "Prefers express shipping", "confidence": 0.9}]'
        )
        facts = extract_facts_from_conversation(SAMPLE_CONVERSATION, api_key="test-key")
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["fact_type"], "order_preference")
        self.assertEqual(facts[0]["fact_text"], "Prefers express shipping")
        self.assertAlmostEqual(facts[0]["confidence"], 0.9)

    def test_multiple_facts_returned(self) -> None:
        from backend.app.fact_extractor import extract_facts_from_conversation

        self._patch_anthropic_with_json(
            '[{"fact_type": "order_preference", "fact_text": "Prefers express shipping", "confidence": 0.9},'
            ' {"fact_type": "issue_history", "fact_text": "Received a damaged item in ORD-1001", "confidence": 0.85}]'
        )
        facts = extract_facts_from_conversation(SAMPLE_CONVERSATION, api_key="test-key")
        self.assertEqual(len(facts), 2)
        fact_types = {f["fact_type"] for f in facts}
        self.assertIn("order_preference", fact_types)
        self.assertIn("issue_history", fact_types)

    def test_low_confidence_facts_filtered_out(self) -> None:
        from backend.app.fact_extractor import extract_facts_from_conversation

        self._patch_anthropic_with_json(
            '[{"fact_type": "order_preference", "fact_text": "Maybe prefers fast shipping", "confidence": 0.4}]'
        )
        facts = extract_facts_from_conversation(SAMPLE_CONVERSATION, api_key="test-key")
        self.assertEqual(facts, [])

    def test_invalid_json_returns_empty_list(self) -> None:
        from backend.app.fact_extractor import extract_facts_from_conversation

        self._patch_anthropic_with_json("not valid json at all")
        facts = extract_facts_from_conversation(SAMPLE_CONVERSATION, api_key="test-key")
        self.assertEqual(facts, [])

    def test_each_fact_has_required_keys(self) -> None:
        from backend.app.fact_extractor import extract_facts_from_conversation

        self._patch_anthropic_with_json(
            '[{"fact_type": "communication_style", "fact_text": "Prefers brief replies", "confidence": 0.8}]'
        )
        facts = extract_facts_from_conversation(SAMPLE_CONVERSATION, api_key="test-key")
        for fact in facts:
            self.assertIn("fact_type", fact)
            self.assertIn("fact_text", fact)
            self.assertIn("confidence", fact)

    def test_no_api_key_returns_empty_list(self) -> None:
        from backend.app.fact_extractor import extract_facts_from_conversation

        facts = extract_facts_from_conversation(SAMPLE_CONVERSATION, api_key=None)
        self.assertEqual(facts, [])


if __name__ == "__main__":
    unittest.main()
