"""Phase 7 — Durable conversation history.

Protocol + InMemoryConversationStore (for tests and no-DB environments).
Each turn is a {role, content} dict compatible with the Claude messages API.
"""

from __future__ import annotations

from typing import Any, Protocol


class ConversationStore(Protocol):
    def save_turn(self, session_id: str, role: str, content: str) -> None: ...

    def load_turns(self, session_id: str, max_turns: int = 20) -> list[dict[str, Any]]: ...


class InMemoryConversationStore:
    """In-process implementation — used in tests and when Postgres is unavailable."""

    def __init__(self) -> None:
        # session_id -> ordered list of {role, content} dicts
        self._turns: dict[str, list[dict[str, Any]]] = {}

    def save_turn(self, session_id: str, role: str, content: str) -> None:
        self._turns.setdefault(session_id, []).append({"role": role, "content": content})

    def load_turns(self, session_id: str, max_turns: int = 20) -> list[dict[str, Any]]:
        all_turns = self._turns.get(session_id, [])
        return [dict(t) for t in all_turns[-max_turns:]]
