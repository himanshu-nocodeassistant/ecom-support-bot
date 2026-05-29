"""Phase 7 — Persistent Customer Memory (TDD: red phase)

Tests are written against the public interface first; implementations follow.
"""

import unittest

from backend.app.conversation_store import InMemoryConversationStore
from backend.app.customer_store import InMemoryCustomerStore
from backend.app.memory_context import build_customer_context, build_system_prompt

# ---------------------------------------------------------------------------
# 7a: Customer identity
# ---------------------------------------------------------------------------


class CustomerIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryCustomerStore()

    def test_upsert_creates_new_customer(self) -> None:
        customer = self.store.upsert_customer(email="alice@example.com", name="Alice")
        self.assertEqual(customer["email"], "alice@example.com")
        self.assertEqual(customer["name"], "Alice")
        self.assertIn("customer_id", customer)

    def test_upsert_returns_same_id_for_existing_email(self) -> None:
        c1 = self.store.upsert_customer(email="alice@example.com", name="Alice")
        c2 = self.store.upsert_customer(email="alice@example.com", name="Alice Updated")
        self.assertEqual(c1["customer_id"], c2["customer_id"])

    def test_different_emails_get_different_ids(self) -> None:
        c1 = self.store.upsert_customer(email="alice@example.com", name="Alice")
        c2 = self.store.upsert_customer(email="bob@example.com", name="Bob")
        self.assertNotEqual(c1["customer_id"], c2["customer_id"])

    def test_link_session_to_customer(self) -> None:
        customer = self.store.upsert_customer(email="alice@example.com", name="Alice")
        self.store.link_session("sess-1", customer["customer_id"])
        linked = self.store.get_customer_id_for_session("sess-1")
        self.assertEqual(linked, customer["customer_id"])

    def test_anonymous_session_returns_none(self) -> None:
        self.assertIsNone(self.store.get_customer_id_for_session("anon-sess-xyz"))

    def test_get_customer_by_id(self) -> None:
        customer = self.store.upsert_customer(email="alice@example.com", name="Alice")
        fetched = self.store.get_customer(customer["customer_id"])
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["email"], "alice@example.com")

    def test_get_unknown_customer_returns_none(self) -> None:
        self.assertIsNone(self.store.get_customer("nonexistent-id"))


# ---------------------------------------------------------------------------
# 7b: Durable conversation history
# ---------------------------------------------------------------------------


class ConversationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryConversationStore()

    def test_save_and_load_turns(self) -> None:
        self.store.save_turn("sess-1", "user", "Hello")
        self.store.save_turn("sess-1", "assistant", "Hi there!")
        turns = self.store.load_turns("sess-1")
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0], {"role": "user", "content": "Hello"})
        self.assertEqual(turns[1], {"role": "assistant", "content": "Hi there!"})

    def test_load_turns_respects_max_limit(self) -> None:
        for i in range(25):
            self.store.save_turn("sess-1", "user", f"msg {i}")
        turns = self.store.load_turns("sess-1", max_turns=20)
        self.assertEqual(len(turns), 20)
        # Should return the most recent 20
        self.assertEqual(turns[-1]["content"], "msg 24")

    def test_empty_session_returns_empty_list(self) -> None:
        self.assertEqual(self.store.load_turns("new-sess"), [])

    def test_different_sessions_are_isolated(self) -> None:
        self.store.save_turn("sess-A", "user", "From session A")
        self.store.save_turn("sess-B", "user", "From session B")
        turns_a = self.store.load_turns("sess-A")
        self.assertEqual(len(turns_a), 1)
        self.assertEqual(turns_a[0]["content"], "From session A")

    def test_load_turns_as_message_dicts(self) -> None:
        self.store.save_turn("sess-1", "user", "What is your return policy?")
        self.store.save_turn("sess-1", "assistant", "We offer 30-day returns.")
        turns = self.store.load_turns("sess-1")
        # Each turn must be usable as a Claude messages entry
        for turn in turns:
            self.assertIn("role", turn)
            self.assertIn("content", turn)
            self.assertIn(turn["role"], ("user", "assistant"))


# ---------------------------------------------------------------------------
# 7c: Memory facts
# ---------------------------------------------------------------------------


class MemoryFactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryCustomerStore()
        self.customer = self.store.upsert_customer(email="alice@example.com", name="Alice")
        self.cid = self.customer["customer_id"]

    def test_save_and_load_memory_fact(self) -> None:
        self.store.save_memory_fact(
            customer_id=self.cid,
            fact_type="issue_history",
            fact_text="Had a delayed order last month",
            confidence=0.9,
            source_session_id="sess-1",
        )
        facts = self.store.load_memory_facts(self.cid)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["fact_text"], "Had a delayed order last month")
        self.assertEqual(facts[0]["fact_type"], "issue_history")

    def test_low_confidence_fact_is_not_stored(self) -> None:
        self.store.save_memory_fact(
            customer_id=self.cid,
            fact_type="order_preference",
            fact_text="Maybe prefers something",
            confidence=0.5,
            source_session_id="sess-1",
        )
        facts = self.store.load_memory_facts(self.cid)
        self.assertEqual(len(facts), 0)

    def test_confidence_threshold_boundary(self) -> None:
        # Exactly 0.7 should be accepted
        self.store.save_memory_fact(
            customer_id=self.cid,
            fact_type="communication_style",
            fact_text="Prefers brief replies",
            confidence=0.7,
            source_session_id="sess-1",
        )
        self.assertEqual(len(self.store.load_memory_facts(self.cid)), 1)

    def test_higher_confidence_overwrites_same_fact_type(self) -> None:
        self.store.save_memory_fact(
            customer_id=self.cid,
            fact_type="order_preference",
            fact_text="Prefers standard shipping",
            confidence=0.75,
            source_session_id="sess-1",
        )
        self.store.save_memory_fact(
            customer_id=self.cid,
            fact_type="order_preference",
            fact_text="Prefers express shipping",
            confidence=0.90,
            source_session_id="sess-2",
        )
        facts = self.store.load_memory_facts(self.cid)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["fact_text"], "Prefers express shipping")

    def test_lower_confidence_does_not_overwrite(self) -> None:
        self.store.save_memory_fact(
            customer_id=self.cid,
            fact_type="order_preference",
            fact_text="Prefers express shipping",
            confidence=0.90,
            source_session_id="sess-1",
        )
        self.store.save_memory_fact(
            customer_id=self.cid,
            fact_type="order_preference",
            fact_text="Prefers standard shipping",
            confidence=0.75,
            source_session_id="sess-2",
        )
        facts = self.store.load_memory_facts(self.cid)
        self.assertEqual(facts[0]["fact_text"], "Prefers express shipping")

    def test_different_fact_types_coexist(self) -> None:
        self.store.save_memory_fact(
            customer_id=self.cid,
            fact_type="issue_history",
            fact_text="Had a delayed order",
            confidence=0.85,
            source_session_id="sess-1",
        )
        self.store.save_memory_fact(
            customer_id=self.cid,
            fact_type="order_preference",
            fact_text="Prefers express shipping",
            confidence=0.80,
            source_session_id="sess-1",
        )
        facts = self.store.load_memory_facts(self.cid)
        self.assertEqual(len(facts), 2)

    def test_facts_for_different_customers_are_isolated(self) -> None:
        c2 = self.store.upsert_customer(email="bob@example.com", name="Bob")
        self.store.save_memory_fact(
            customer_id=self.cid,
            fact_type="issue_history",
            fact_text="Alice had a delay",
            confidence=0.85,
            source_session_id="sess-1",
        )
        self.assertEqual(len(self.store.load_memory_facts(c2["customer_id"])), 0)


# ---------------------------------------------------------------------------
# 7e: Order linkage
# ---------------------------------------------------------------------------


class OrderLinkageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryCustomerStore()
        self.customer = self.store.upsert_customer(email="alice@example.com", name="Alice")
        self.cid = self.customer["customer_id"]

    def test_link_order_to_customer(self) -> None:
        self.store.link_order(self.cid, "ORD-1001")
        orders = self.store.get_customer_orders(self.cid)
        self.assertIn("ORD-1001", orders)

    def test_multiple_orders_stored(self) -> None:
        self.store.link_order(self.cid, "ORD-1001")
        self.store.link_order(self.cid, "ORD-1002")
        orders = self.store.get_customer_orders(self.cid)
        self.assertIn("ORD-1001", orders)
        self.assertIn("ORD-1002", orders)

    def test_duplicate_order_not_added_twice(self) -> None:
        self.store.link_order(self.cid, "ORD-1001")
        self.store.link_order(self.cid, "ORD-1001")
        orders = self.store.get_customer_orders(self.cid)
        self.assertEqual(orders.count("ORD-1001"), 1)

    def test_orders_capped_at_five(self) -> None:
        for i in range(7):
            self.store.link_order(self.cid, f"ORD-200{i}")
        orders = self.store.get_customer_orders(self.cid)
        self.assertLessEqual(len(orders), 5)

    def test_no_orders_for_unknown_customer(self) -> None:
        self.assertEqual(self.store.get_customer_orders("nonexistent"), [])


# ---------------------------------------------------------------------------
# 7d: Memory context building
# ---------------------------------------------------------------------------


class MemoryContextTests(unittest.TestCase):
    def test_context_includes_facts(self) -> None:
        facts = [
            {"fact_type": "issue_history", "fact_text": "Had a delayed order last month"},
            {"fact_type": "order_preference", "fact_text": "Prefers express shipping"},
        ]
        ctx = build_customer_context(facts=facts, prior_order_ids=["ORD-1001"])
        self.assertIn("Had a delayed order last month", ctx)
        self.assertIn("Prefers express shipping", ctx)

    def test_context_includes_prior_order_ids(self) -> None:
        ctx = build_customer_context(facts=[], prior_order_ids=["ORD-1001", "ORD-1002"])
        self.assertIn("ORD-1001", ctx)
        self.assertIn("ORD-1002", ctx)

    def test_empty_facts_and_orders_returns_empty_string(self) -> None:
        ctx = build_customer_context(facts=[], prior_order_ids=[])
        self.assertEqual(ctx, "")

    def test_context_only_orders_no_facts(self) -> None:
        ctx = build_customer_context(facts=[], prior_order_ids=["ORD-1001"])
        self.assertIn("ORD-1001", ctx)
        self.assertNotEqual(ctx, "")

    def test_system_prompt_contains_base_instructions(self) -> None:
        prompt = build_system_prompt(customer_context="")
        self.assertIn("lookup_order", prompt)
        self.assertIn("search_knowledge_base", prompt)

    def test_system_prompt_prepends_customer_context(self) -> None:
        ctx = "## Customer context\n- Had a delayed order"
        prompt = build_system_prompt(customer_context=ctx)
        self.assertIn("Customer context", prompt)
        self.assertIn("Had a delayed order", prompt)
        # Context should appear before the base instructions
        ctx_pos = prompt.index("Customer context")
        tool_pos = prompt.index("lookup_order")
        self.assertLess(ctx_pos, tool_pos)

    def test_system_prompt_without_context_matches_base(self) -> None:
        prompt_no_ctx = build_system_prompt(customer_context="")
        # Should still contain the agent instructions unchanged
        self.assertIn("create_ticket", prompt_no_ctx)


if __name__ == "__main__":
    unittest.main()
