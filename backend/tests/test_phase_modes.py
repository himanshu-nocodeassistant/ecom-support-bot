"""Tests for phase-aware execution modes and the /api/compare endpoint."""

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from backend.app.agent import SESSION_MEMORY, _handle_message_deterministic, handle_compare
from backend.app.data import KNOWLEDGE_BASE, ORDERS
from backend.app.repository import InMemoryRepository


def _in_memory_repo() -> InMemoryRepository:
    return InMemoryRepository(orders=ORDERS, knowledge_documents=KNOWLEDGE_BASE)


class Phase1ModeTests(unittest.TestCase):
    """Phase 1 uses keyword-only InMemoryRepository regardless of env config."""

    def setUp(self) -> None:
        SESSION_MEMORY.clear()
        self._repo = _in_memory_repo()

    def test_knowledge_query_keyword_match(self) -> None:
        result = _handle_message_deterministic(
            "s1", "Does the blender have a safety lock?", mode="phase1"
        )
        self.assertEqual(result["mode"], "phase1")
        names = [e["name"] for e in result["tool_events"]]
        self.assertIn("search_knowledge_base", names)
        self.assertIn("blender", result["reply"].lower())

    def test_semantic_miss_escalates_to_ticket(self) -> None:
        # No token from this query matches any KB document — phase1 must escalate
        result = _handle_message_deterministic(
            "s2", "Seeking reimbursement for faulty apparatus", mode="phase1"
        )
        names = [e["name"] for e in result["tool_events"]]
        self.assertIn("create_ticket", names)

    def test_order_lookup_uses_in_memory_orders(self) -> None:
        result = _handle_message_deterministic("s3", "Where is my order ORD-1001?", mode="phase1")
        names = [e["name"] for e in result["tool_events"]]
        self.assertIn("lookup_order", names)
        self.assertIn("ORD-1001", result["reply"])

    def test_refund_for_delivered_order(self) -> None:
        result = _handle_message_deterministic("s4", "I want a refund for ORD-1002", mode="phase1")
        names = [e["name"] for e in result["tool_events"]]
        self.assertIn("request_refund", names)

    def test_refund_for_undelivered_order_is_rejected(self) -> None:
        result = _handle_message_deterministic("s5", "refund for ORD-1001", mode="phase1")
        refund_evt = next((e for e in result["tool_events"] if e["name"] == "request_refund"), None)
        self.assertIsNotNone(refund_evt)
        self.assertFalse(refund_evt["output"]["approved"])

    def test_result_includes_mode_field(self) -> None:
        result = _handle_message_deterministic("s6", "hello", mode="phase1")
        self.assertEqual(result["mode"], "phase1")


class Phase2ModeTests(unittest.TestCase):
    """Phase 2 uses the same deterministic routing as Phase 1 but routes to PostgresRepository
    when available. In the test environment without a real DB it falls back to InMemoryRepository,
    so the behaviour difference is only in the repository used — we test the mode field and that
    the semantic query path is attempted."""

    def setUp(self) -> None:
        SESSION_MEMORY.clear()

    def test_mode_field_is_phase2(self) -> None:
        result = _handle_message_deterministic("s7", "Can I get my money back?", mode="phase2")
        self.assertEqual(result["mode"], "phase2")

    def test_knowledge_search_is_called(self) -> None:
        result = _handle_message_deterministic(
            "s8", "Does the blender have a safety lock?", mode="phase2"
        )
        names = [e["name"] for e in result["tool_events"]]
        self.assertIn("search_knowledge_base", names)


class Phase3ModeTests(unittest.TestCase):
    """Phase 3 uses the Claude agent tool loop (mocked here)."""

    def setUp(self) -> None:
        SESSION_MEMORY.clear()
        self._repo = _in_memory_repo()
        self._repo_patcher = patch("backend.app.agent.get_repository", return_value=self._repo)
        self._repo_patcher.start()
        self._settings_patcher = patch(
            "backend.app.agent.get_settings",
            return_value=MagicMock(
                anthropic_api_key="test-key",
                data_backend="memory",
                database_url=None,
                voyage_api_key=None,
            ),
        )
        self._settings_patcher.start()

    def tearDown(self) -> None:
        self._repo_patcher.stop()
        self._settings_patcher.stop()

    def _patch_anthropic(self, side_effects):
        client = MagicMock()
        client.messages.create.side_effect = side_effects
        patcher = patch("anthropic.Anthropic", return_value=client)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _text_block(self, text):
        b = MagicMock()
        b.type = "text"
        b.text = text
        return b

    def _tool_use_block(self, tool_id, name, inputs):
        b = MagicMock()
        b.type = "tool_use"
        b.id = tool_id
        b.name = name
        b.input = inputs
        return b

    def _response(self, stop_reason, *blocks):
        r = MagicMock()
        r.stop_reason = stop_reason
        r.content = list(blocks)
        return r

    def test_phase3_mode_uses_agent_loop(self) -> None:
        from backend.app.agent import handle_message

        self._patch_anthropic(
            [
                self._response(
                    "tool_use", self._tool_use_block("t1", "lookup_order", {"order_id": "ORD-1001"})
                ),
                self._response("end_turn", self._text_block("Order ORD-1001 is shipped.")),
            ]
        )

        result = handle_message("s9", "Where is ORD-1001?", mode="phase3")
        self.assertEqual(result["mode"], "phase3")
        self.assertIn("ORD-1001", result["reply"])

    def test_phase3_result_has_mode_field(self) -> None:
        from backend.app.agent import handle_message

        self._patch_anthropic(
            [
                self._response("end_turn", self._text_block("Hello!")),
            ]
        )

        result = handle_message("s10", "hi", mode="phase3")
        self.assertEqual(result["mode"], "phase3")


class CompareEndpointTests(unittest.TestCase):
    """handle_compare runs phase1, phase2, phase3 concurrently and returns all results."""

    def setUp(self) -> None:
        SESSION_MEMORY.clear()

    def _settings_no_key(self):
        return MagicMock(
            anthropic_api_key=None,
            data_backend="memory",
            database_url=None,
            voyage_api_key=None,
        )

    def test_compare_returns_all_three_phases(self) -> None:
        with patch("backend.app.agent.get_settings", return_value=self._settings_no_key()):
            result = asyncio.run(handle_compare("Does the blender have a safety lock?"))

        self.assertIn("phase1", result)
        self.assertIn("phase2", result)
        self.assertIn("phase3", result)

    def test_compare_phase_fields_contain_reply(self) -> None:
        with patch("backend.app.agent.get_settings", return_value=self._settings_no_key()):
            result = asyncio.run(handle_compare("Where is my order ORD-1001?"))

        for phase in ("phase1", "phase2", "phase3"):
            self.assertIn("reply", result[phase])
            self.assertIsInstance(result[phase]["reply"], str)

    def test_compare_phase_fields_contain_tool_events(self) -> None:
        with patch("backend.app.agent.get_settings", return_value=self._settings_no_key()):
            result = asyncio.run(handle_compare("Where is my order ORD-1001?"))

        for phase in ("phase1", "phase2", "phase3"):
            self.assertIn("tool_events", result[phase])
            self.assertIsInstance(result[phase]["tool_events"], list)

    def test_compare_mode_fields_are_correct(self) -> None:
        with patch("backend.app.agent.get_settings", return_value=self._settings_no_key()):
            result = asyncio.run(handle_compare("hello"))

        self.assertEqual(result["phase1"]["mode"], "phase1")
        self.assertEqual(result["phase2"]["mode"], "phase2")
        # phase3 falls back to deterministic when no API key, mode field still present
        self.assertIn("mode", result["phase3"])


class SSEStreamingTests(unittest.TestCase):
    """handle_message_stream yields well-formed SSE events."""

    def setUp(self) -> None:
        SESSION_MEMORY.clear()

    def _settings_no_key(self):
        return MagicMock(
            anthropic_api_key=None,
            data_backend="memory",
            database_url=None,
            voyage_api_key=None,
        )

    def _collect(self, gen):
        results = []

        async def _run():
            async for chunk in gen:
                results.append(chunk)

        asyncio.run(_run())
        return results

    def _parse_events(self, chunks):
        events = []
        for chunk in chunks:
            lines = chunk.strip().split("\n")
            event_type = None
            data = None
            for line in lines:
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                elif line.startswith("data: "):
                    import json

                    data = json.loads(line[6:].strip())
            if event_type:
                events.append((event_type, data))
        return events

    def test_stream_emits_token_and_done(self) -> None:
        from backend.app.agent import handle_message_stream

        with patch("backend.app.agent.get_settings", return_value=self._settings_no_key()):
            chunks = self._collect(
                handle_message_stream("stream-1", "Does the blender have a safety lock?")
            )

        events = self._parse_events(chunks)
        event_types = [e[0] for e in events]
        self.assertIn("token", event_types)
        self.assertIn("done", event_types)
        # done must be last
        self.assertEqual(event_types[-1], "done")

    def test_stream_done_event_has_session_id(self) -> None:
        from backend.app.agent import handle_message_stream

        with patch("backend.app.agent.get_settings", return_value=self._settings_no_key()):
            chunks = self._collect(handle_message_stream("stream-2", "hello"))

        events = self._parse_events(chunks)
        done_events = [(t, d) for t, d in events if t == "done"]
        self.assertTrue(len(done_events) >= 1)
        self.assertEqual(done_events[0][1]["session_id"], "stream-2")

    def test_stream_sse_format(self) -> None:
        from backend.app.agent import handle_message_stream

        with patch("backend.app.agent.get_settings", return_value=self._settings_no_key()):
            chunks = self._collect(handle_message_stream("stream-3", "hello"))

        for chunk in chunks:
            self.assertIn("event:", chunk)
            self.assertIn("data:", chunk)
            self.assertTrue(chunk.endswith("\n\n"))


if __name__ == "__main__":
    unittest.main()
