"""Phase 7 — Customer identity, memory facts, and order linkage.

Protocol + InMemoryCustomerStore (for tests and no-DB environments).
PostgresCustomerStore added later once the interface is stable.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

_CONFIDENCE_THRESHOLD = 0.7
_MAX_LINKED_ORDERS = 5
_TTL_DAYS = 90


def _now() -> datetime:
    return datetime.now(tz=UTC)


class CustomerStore(Protocol):
    def upsert_customer(self, email: str, name: str) -> dict[str, Any]: ...

    def get_customer(self, customer_id: str) -> dict[str, Any] | None: ...

    def link_session(self, session_id: str, customer_id: str) -> None: ...

    def get_customer_id_for_session(self, session_id: str) -> str | None: ...

    def save_memory_fact(
        self,
        customer_id: str,
        fact_type: str,
        fact_text: str,
        confidence: float,
        source_session_id: str,
    ) -> None: ...

    def load_memory_facts(self, customer_id: str) -> list[dict[str, Any]]: ...

    def link_order(self, customer_id: str, order_id: str) -> None: ...

    def get_customer_orders(self, customer_id: str) -> list[str]: ...


class InMemoryCustomerStore:
    """In-process implementation — used in tests and when Postgres is unavailable."""

    def __init__(self) -> None:
        # email -> customer dict
        self._by_email: dict[str, dict[str, Any]] = {}
        # customer_id -> customer dict
        self._by_id: dict[str, dict[str, Any]] = {}
        # session_id -> customer_id
        self._sessions: dict[str, str] = {}
        # customer_id -> {fact_type -> fact dict}
        self._facts: dict[str, dict[str, dict[str, Any]]] = {}
        # customer_id -> list[order_id] (most recent first, max 5)
        self._orders: dict[str, list[str]] = {}

    def upsert_customer(self, email: str, name: str) -> dict[str, Any]:
        if email in self._by_email:
            return dict(self._by_email[email])
        customer: dict[str, Any] = {
            "customer_id": str(uuid.uuid4()),
            "email": email,
            "name": name,
        }
        self._by_email[email] = customer
        self._by_id[customer["customer_id"]] = customer
        return dict(customer)

    def get_customer(self, customer_id: str) -> dict[str, Any] | None:
        row = self._by_id.get(customer_id)
        return dict(row) if row else None

    def link_session(self, session_id: str, customer_id: str) -> None:
        self._sessions[session_id] = customer_id

    def get_customer_id_for_session(self, session_id: str) -> str | None:
        return self._sessions.get(session_id)

    def save_memory_fact(
        self,
        customer_id: str,
        fact_type: str,
        fact_text: str,
        confidence: float,
        source_session_id: str,
    ) -> None:
        if confidence < _CONFIDENCE_THRESHOLD:
            return
        bucket = self._facts.setdefault(customer_id, {})
        existing = bucket.get(fact_type)
        if existing is None or confidence >= existing["confidence"]:
            bucket[fact_type] = {
                "fact_type": fact_type,
                "fact_text": fact_text,
                "confidence": confidence,
                "source_session_id": source_session_id,
                "expires_at": _now() + timedelta(days=_TTL_DAYS),
            }

    def load_memory_facts(self, customer_id: str) -> list[dict[str, Any]]:
        bucket = self._facts.get(customer_id, {})
        now = _now()
        return [
            {k: v for k, v in f.items() if k != "expires_at"}
            for f in bucket.values()
            if f.get("expires_at", now) >= now
        ]

    def link_order(self, customer_id: str, order_id: str) -> None:
        orders = self._orders.setdefault(customer_id, [])
        if order_id in orders:
            return
        orders.insert(0, order_id)
        self._orders[customer_id] = orders[:_MAX_LINKED_ORDERS]

    def get_customer_orders(self, customer_id: str) -> list[str]:
        return list(self._orders.get(customer_id, []))


class PostgresCustomerStore:
    """Postgres-backed customer store.

    Requires psycopg2. Instantiated only when DATABASE_URL is set; the caller
    is responsible for falling back to InMemoryCustomerStore when unavailable.
    """

    def __init__(self, database_url: str) -> None:
        import psycopg2
        import psycopg2.extras

        self._conn = psycopg2.connect(database_url)
        self._conn.autocommit = True

    def upsert_customer(self, email: str, name: str) -> dict[str, Any]:
        import psycopg2.extras

        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO customers (email, name)
                VALUES (%s, %s)
                ON CONFLICT (email) DO UPDATE SET name = EXCLUDED.name
                RETURNING customer_id, email, name
                """,
                (email, name or ""),
            )
            row = cur.fetchone()
        return dict(row)

    def get_customer(self, customer_id: str) -> dict[str, Any] | None:
        import psycopg2.extras

        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT customer_id, email, name FROM customers WHERE customer_id = %s",
                (customer_id,),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def link_session(self, session_id: str, customer_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO customer_sessions (session_id, customer_id)
                VALUES (%s, %s)
                ON CONFLICT (session_id) DO UPDATE SET last_active_at = now()
                """,
                (session_id, customer_id),
            )

    def get_customer_id_for_session(self, session_id: str) -> str | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT customer_id FROM customer_sessions WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
        return str(row[0]) if row else None

    def save_memory_fact(
        self,
        customer_id: str,
        fact_type: str,
        fact_text: str,
        confidence: float,
        source_session_id: str,
    ) -> None:
        if confidence < _CONFIDENCE_THRESHOLD:
            return
        expires_at = _now() + timedelta(days=_TTL_DAYS)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO customer_memory
                    (customer_id, fact_type, fact_text, confidence, source_session_id, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (customer_id, fact_type)
                DO UPDATE SET
                    fact_text = EXCLUDED.fact_text,
                    confidence = EXCLUDED.confidence,
                    source_session_id = EXCLUDED.source_session_id,
                    expires_at = EXCLUDED.expires_at,
                    updated_at = now()
                WHERE customer_memory.confidence <= EXCLUDED.confidence
                """,
                (customer_id, fact_type, fact_text, confidence, source_session_id, expires_at),
            )

    def load_memory_facts(self, customer_id: str) -> list[dict[str, Any]]:
        import psycopg2.extras

        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT fact_type, fact_text, confidence, source_session_id
                FROM customer_memory
                WHERE customer_id = %s AND expires_at >= now()
                """,
                (customer_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def link_order(self, customer_id: str, order_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO customer_orders (customer_id, order_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
                """,
                (customer_id, order_id),
            )
            cur.execute(
                """
                DELETE FROM customer_orders
                WHERE customer_id = %s
                  AND order_id NOT IN (
                      SELECT order_id FROM customer_orders
                      WHERE customer_id = %s
                      ORDER BY linked_at DESC
                      LIMIT %s
                  )
                """,
                (customer_id, customer_id, _MAX_LINKED_ORDERS),
            )

    def get_customer_orders(self, customer_id: str) -> list[str]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT order_id FROM customer_orders
                WHERE customer_id = %s
                ORDER BY linked_at DESC
                LIMIT %s
                """,
                (customer_id, _MAX_LINKED_ORDERS),
            )
            return [row[0] for row in cur.fetchall()]
