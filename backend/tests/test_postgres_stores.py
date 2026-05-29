"""Phase 8d — Postgres store existence + 90-day TTL on memory facts (TDD: red phase).

8d-i:  PostgresConversationStore and PostgresCustomerStore classes exist and implement
       the same protocol as their in-memory counterparts.
8d-ii: InMemoryCustomerStore (and Postgres counterpart) respects a 90-day TTL:
       expired facts are not returned by load_memory_facts().
"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from backend.app.customer_store import InMemoryCustomerStore

# ---------------------------------------------------------------------------
# 8d-i: Postgres store classes exist
# ---------------------------------------------------------------------------


class PostgresStoreExistenceTests(unittest.TestCase):
    def test_postgres_conversation_store_importable(self) -> None:
        from backend.app.conversation_store import PostgresConversationStore  # noqa

    def test_postgres_customer_store_importable(self) -> None:
        from backend.app.customer_store import PostgresCustomerStore  # noqa

    def test_postgres_conversation_store_has_save_turn(self) -> None:
        from backend.app.conversation_store import PostgresConversationStore

        self.assertTrue(hasattr(PostgresConversationStore, "save_turn"))

    def test_postgres_conversation_store_has_load_turns(self) -> None:
        from backend.app.conversation_store import PostgresConversationStore

        self.assertTrue(hasattr(PostgresConversationStore, "load_turns"))

    def test_postgres_customer_store_has_upsert_customer(self) -> None:
        from backend.app.customer_store import PostgresCustomerStore

        self.assertTrue(hasattr(PostgresCustomerStore, "upsert_customer"))

    def test_postgres_customer_store_has_save_memory_fact(self) -> None:
        from backend.app.customer_store import PostgresCustomerStore

        self.assertTrue(hasattr(PostgresCustomerStore, "save_memory_fact"))


# ---------------------------------------------------------------------------
# 8d-ii: 90-day TTL enforced on memory facts
# ---------------------------------------------------------------------------


class MemoryFactTTLTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryCustomerStore()
        self.customer = self.store.upsert_customer("alice@example.com", "Alice")
        self.cid = self.customer["customer_id"]

    def test_fresh_fact_is_returned(self) -> None:
        self.store.save_memory_fact(
            customer_id=self.cid,
            fact_type="order_preference",
            fact_text="Prefers express shipping",
            confidence=0.9,
            source_session_id="sess-1",
        )
        facts = self.store.load_memory_facts(self.cid)
        self.assertEqual(len(facts), 1)

    def test_expired_fact_is_not_returned(self) -> None:
        self.store.save_memory_fact(
            customer_id=self.cid,
            fact_type="order_preference",
            fact_text="Prefers express shipping",
            confidence=0.9,
            source_session_id="sess-1",
        )
        # Travel 91 days into the future
        future = datetime.now(tz=UTC) + timedelta(days=91)
        with patch("backend.app.customer_store._now", return_value=future):
            facts = self.store.load_memory_facts(self.cid)
        self.assertEqual(facts, [])

    def test_fact_at_boundary_90_days_still_returned(self) -> None:
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        with patch("backend.app.customer_store._now", return_value=t0):
            self.store.save_memory_fact(
                customer_id=self.cid,
                fact_type="order_preference",
                fact_text="Prefers express shipping",
                confidence=0.9,
                source_session_id="sess-1",
            )
        # Exactly at expiry — still valid (boundary is inclusive)
        boundary = t0 + timedelta(days=90)
        with patch("backend.app.customer_store._now", return_value=boundary):
            facts = self.store.load_memory_facts(self.cid)
        self.assertEqual(len(facts), 1)

    def test_expired_and_fresh_facts_mixed(self) -> None:
        self.store.save_memory_fact(
            customer_id=self.cid,
            fact_type="issue_history",
            fact_text="Had a damaged item",
            confidence=0.85,
            source_session_id="sess-old",
        )
        # Advance time past expiry
        future = datetime.now(tz=UTC) + timedelta(days=91)
        with patch("backend.app.customer_store._now", return_value=future):
            # Save a new fact — its expires_at is 91 days + 90 days from now
            self.store.save_memory_fact(
                customer_id=self.cid,
                fact_type="order_preference",
                fact_text="Prefers express shipping",
                confidence=0.9,
                source_session_id="sess-new",
            )
            facts = self.store.load_memory_facts(self.cid)

        # The old issue_history fact should be expired, the new one should be fresh
        fact_types = {f["fact_type"] for f in facts}
        self.assertIn("order_preference", fact_types)
        self.assertNotIn("issue_history", fact_types)


if __name__ == "__main__":
    unittest.main()
