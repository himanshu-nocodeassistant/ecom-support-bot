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


class PostgresConversationStore:
    """Postgres-backed conversation store.

    Requires psycopg2. Instantiated only when DATABASE_URL is set; the caller
    is responsible for falling back to InMemoryConversationStore when unavailable.
    """

    def __init__(self, database_url: str) -> None:
        import psycopg2

        self._conn = psycopg2.connect(database_url)
        self._conn.autocommit = True

    def save_turn(self, session_id: str, role: str, content: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversation_turns (session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, role, content),
            )

    def load_turns(self, session_id: str, max_turns: int = 20) -> list[dict[str, Any]]:
        import psycopg2.extras

        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT role, content FROM (
                    SELECT role, content, created_at
                    FROM conversation_turns
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ) sub ORDER BY created_at ASC
                """,
                (session_id, max_turns),
            )
            return [{"role": row["role"], "content": row["content"]} for row in cur.fetchall()]
