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


# ---------------------------------------------------------------------------
# Gap 3 — Cross-turn order recall (af09)
# ---------------------------------------------------------------------------


class CrossTurnOrderRecallTests(unittest.TestCase):
    """Gap 3 / af09: the agent must recall an order ID from an earlier turn without re-identification."""

    def setUp(self) -> None:
        from backend.app.agent import SESSION_MEMORY

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

    def _text_block(self, text: str) -> MagicMock:
        b = MagicMock()
        b.type = "text"
        b.text = text
        return b

    def _tool_use_block(self, tool_id: str, name: str, inputs: dict) -> MagicMock:
        b = MagicMock()
        b.type = "tool_use"
        b.id = tool_id
        b.name = name
        b.input = inputs
        return b

    def _response(self, stop_reason: str, *blocks: MagicMock) -> MagicMock:
        r = MagicMock()
        r.stop_reason = stop_reason
        r.content = list(blocks)
        return r

    def test_agent_recalls_order_id_from_earlier_turn(self) -> None:
        from backend.app.agent import handle_message

        side_effects = [
            # Turn 1: look up ORD-1002, reply with shipping info
            self._response(
                "tool_use",
                self._tool_use_block("t1", "lookup_order", {"order_id": "ORD-1002"}),
            ),
            self._response("end_turn", self._text_block("ORD-1002 shipped on 2026-05-12.")),
            # Turn 2: agent remembers ORD-1002 from context, processes return/refund
            self._response(
                "tool_use",
                self._tool_use_block("t2", "lookup_order", {"order_id": "ORD-1002"}),
            ),
            self._response(
                "tool_use",
                self._tool_use_block(
                    "t3",
                    "request_refund",
                    {"order_id": "ORD-1002", "reason": "customer wants to return"},
                ),
            ),
            self._response("end_turn", self._text_block("Refund for ORD-1002 created.")),
        ]

        with patch(
            "anthropic.Anthropic",
            return_value=MagicMock(messages=MagicMock(create=MagicMock(side_effect=side_effects))),
        ):
            handle_message("af09-unit", "My order is ORD-1002. When did it ship?", mode="phase3")
            result = handle_message("af09-unit", "And can I return it?", mode="phase3")

        all_tool_names = [e["name"] for e in result["tool_events"]]
        # The second turn must involve ORD-1002 — either via lookup or refund
        self.assertTrue(
            "lookup_order" in all_tool_names or "request_refund" in all_tool_names,
            f"Turn 2 must call an order-related tool to handle the return. Got: {all_tool_names}",
        )

    def test_session_memory_persists_full_history_across_turns(self) -> None:
        from backend.app.agent import SESSION_MEMORY, handle_message

        side_effects = [
            self._response(
                "tool_use",
                self._tool_use_block("t1", "lookup_order", {"order_id": "ORD-1001"}),
            ),
            self._response("end_turn", self._text_block("ORD-1001 is in transit.")),
            self._response("end_turn", self._text_block("Standard delivery takes 3-7 days.")),
        ]
        client_mock = MagicMock()
        client_mock.messages.create.side_effect = side_effects

        with patch("anthropic.Anthropic", return_value=client_mock):
            handle_message("persist-test", "Where is ORD-1001?", mode="phase3")
            handle_message("persist-test", "How long does delivery take?", mode="phase3")

        # SESSION_MEMORY must contain multiple turns for cross-turn recall to work
        history = SESSION_MEMORY.get("persist-test", [])
        self.assertGreater(len(history), 2, "Session memory must retain full multi-turn history")


# ---------------------------------------------------------------------------
# Gap 4 — Fact extraction fires after session turn
# ---------------------------------------------------------------------------


class FactExtractionBackgroundTaskTests(unittest.TestCase):
    """Gap 4: run_fact_extraction must read from SESSION_MEMORY and persist facts."""

    def setUp(self) -> None:
        from backend.app.agent import SESSION_MEMORY

        SESSION_MEMORY.clear()

    def test_run_fact_extraction_reads_session_memory_and_calls_extractor(self) -> None:
        from unittest.mock import patch

        from backend.app.agent import SESSION_MEMORY
        from backend.app.main import run_fact_extraction

        # Populate SESSION_MEMORY with a conversation mentioning a preference
        SESSION_MEMORY["fe-test"] = [
            {"role": "user", "content": "I always use express shipping."},
            {
                "role": "assistant",
                "content": "Noted! I'll remember your preference for express shipping.",
            },
            {"role": "user", "content": "Can you check my order?"},
            {"role": "assistant", "content": "Sure, what is your order ID?"},
        ]

        captured_convos = []

        def fake_extract(conversation, api_key):
            captured_convos.append(conversation)
            return [
                {
                    "fact_type": "order_preference",
                    "fact_text": "Prefers express shipping",
                    "confidence": 0.9,
                }
            ]

        with (
            patch(
                "backend.app.config.get_settings", return_value=MagicMock(anthropic_api_key="test")
            ),
            patch("backend.app.fact_extractor.extract_facts_from_conversation", fake_extract),
        ):
            run_fact_extraction("fe-test", "alice@example.com")

        self.assertEqual(
            len(captured_convos), 1, "extract_facts_from_conversation must be called once"
        )
        convo = captured_convos[0]
        # Only string-content messages should be passed
        self.assertTrue(all(isinstance(m["content"], str) for m in convo))

    def test_run_fact_extraction_skips_when_no_session_history(self) -> None:
        from unittest.mock import patch

        from backend.app.main import run_fact_extraction

        called = []

        def fake_extract(conversation, api_key):
            called.append(True)
            return []

        with (
            patch(
                "backend.app.config.get_settings", return_value=MagicMock(anthropic_api_key="test")
            ),
            patch("backend.app.fact_extractor.extract_facts_from_conversation", fake_extract),
        ):
            run_fact_extraction("nonexistent-session", "alice@example.com")

        self.assertEqual(len(called), 0, "extractor must not be called for empty session")

    def test_chat_endpoint_uses_background_tasks(self) -> None:
        import inspect

        from backend.app.main import chat

        sig = inspect.signature(chat)
        self.assertIn(
            "background_tasks",
            sig.parameters,
            "/api/chat endpoint must accept a BackgroundTasks parameter",
        )


if __name__ == "__main__":
    unittest.main()
