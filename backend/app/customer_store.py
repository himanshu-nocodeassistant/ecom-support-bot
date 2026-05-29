"""Phase 7 — Customer identity, memory facts, and order linkage.

Protocol + InMemoryCustomerStore (for tests and no-DB environments).
PostgresCustomerStore added later once the interface is stable.
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol

_CONFIDENCE_THRESHOLD = 0.7
_MAX_LINKED_ORDERS = 5


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
            }

    def load_memory_facts(self, customer_id: str) -> list[dict[str, Any]]:
        bucket = self._facts.get(customer_id, {})
        return [dict(f) for f in bucket.values()]

    def link_order(self, customer_id: str, order_id: str) -> None:
        orders = self._orders.setdefault(customer_id, [])
        if order_id in orders:
            return
        orders.insert(0, order_id)
        self._orders[customer_id] = orders[:_MAX_LINKED_ORDERS]

    def get_customer_orders(self, customer_id: str) -> list[str]:
        return list(self._orders.get(customer_id, []))
