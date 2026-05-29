"""Phase 8e — API customer_email field (TDD: red phase).

/api/chat and /api/chat/stream must accept an optional customer_email field
and pass it through to the agent handlers.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.app.main import app


def _text_block(text: str) -> MagicMock:
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _response(stop_reason: str, *blocks: MagicMock) -> MagicMock:
    r = MagicMock()
    r.stop_reason = stop_reason
    r.content = list(blocks)
    return r


class ChatEndpointCustomerEmailTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self._settings_patcher = patch(
            "backend.app.agent.get_settings",
            return_value=MagicMock(anthropic_api_key="test-key"),
        )
        self._settings_patcher.start()

    def tearDown(self) -> None:
        self._settings_patcher.stop()

    def _patch_anthropic(self, side_effects: list[MagicMock]) -> None:
        client = MagicMock()
        client.messages.create.side_effect = side_effects
        patcher = patch("anthropic.Anthropic", return_value=client)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_chat_accepts_customer_email_field(self) -> None:
        self._patch_anthropic([_response("end_turn", _text_block("Hi!"))])

        resp = self.client.post(
            "/api/chat",
            json={
                "session_id": "sess-1",
                "message": "Hello",
                "customer_email": "alice@example.com",
            },
        )
        self.assertEqual(resp.status_code, 200)

    def test_chat_works_without_customer_email(self) -> None:
        self._patch_anthropic([_response("end_turn", _text_block("Hi!"))])

        resp = self.client.post(
            "/api/chat",
            json={"session_id": "sess-1", "message": "Hello"},
        )
        self.assertEqual(resp.status_code, 200)

    def test_chat_with_customer_email_returns_reply(self) -> None:
        self._patch_anthropic([_response("end_turn", _text_block("Hello Alice!"))])

        resp = self.client.post(
            "/api/chat",
            json={
                "session_id": "sess-1",
                "message": "Hi",
                "customer_email": "alice@example.com",
            },
        )
        body = resp.json()
        self.assertEqual(body["reply"], "Hello Alice!")


class StreamEndpointCustomerEmailTests(unittest.TestCase):
    """SSE stream endpoint must accept customer_email and pass it to the streaming handler."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        self._settings_patcher = patch(
            "backend.app.agent.get_settings",
            return_value=MagicMock(anthropic_api_key="test-key"),
        )
        self._settings_patcher.start()

    def tearDown(self) -> None:
        self._settings_patcher.stop()

    def _stream_events(self, payload: dict) -> list[dict]:
        """Collect SSE events from a stream request."""
        with self.client.stream("POST", "/api/chat/stream", json=payload) as resp:
            self.assertEqual(resp.status_code, 200)
            events = []
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:].strip()))
            return events

    def test_stream_accepts_customer_email_field(self) -> None:
        async def _fake_stream(session_id, message, **kwargs):
            yield 'event: token\ndata: {"text": "Hi!"}\n\n'
            yield 'event: done\ndata: {"session_id": "' + session_id + '", "tool_count": 0}\n\n'

        with patch("backend.app.main.handle_message_stream", side_effect=_fake_stream):
            resp = self.client.post(
                "/api/chat/stream",
                json={
                    "session_id": "sess-2",
                    "message": "Hello",
                    "customer_email": "alice@example.com",
                },
            )
        self.assertEqual(resp.status_code, 200)

    def test_stream_customer_email_forwarded_to_handler(self) -> None:
        """The customer_email from the request body reaches handle_message_stream."""
        received_kwargs: dict = {}

        async def _capture_stream(session_id, message, **kwargs):
            received_kwargs.update(kwargs)
            yield 'event: done\ndata: {"session_id": "' + session_id + '", "tool_count": 0}\n\n'

        with patch("backend.app.main.handle_message_stream", side_effect=_capture_stream):
            self.client.post(
                "/api/chat/stream",
                json={
                    "session_id": "sess-3",
                    "message": "Hi",
                    "customer_email": "bob@example.com",
                },
            )

        self.assertEqual(received_kwargs.get("customer_email"), "bob@example.com")


if __name__ == "__main__":
    unittest.main()
