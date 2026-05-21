import unittest

from backend.app.agent import handle_message


class SupportBotTests(unittest.TestCase):
    def test_product_question_uses_kb(self) -> None:
        result = handle_message("session-product", "Does the portable blender have a safety lock?")
        self.assertIn("portable blender", result["reply"].lower())
        self.assertEqual(result["tool_events"][0]["name"], "search_knowledge_base")

    def test_order_status_lookup(self) -> None:
        result = handle_message("session-order", "Where is my order ORD-1001?")
        self.assertIn("ORD-1001", result["reply"])
        self.assertEqual(result["tool_events"][0]["name"], "lookup_order")

    def test_memory_recalls_order_id_for_follow_up(self) -> None:
        handle_message("session-memory", "Where is my order ORD-1002?")
        result = handle_message("session-memory", "Can I get a refund for it?")
        self.assertEqual(result["tool_events"][0]["name"], "request_refund")
        self.assertIn("Refund request created", result["reply"])

    def test_low_confidence_creates_ticket(self) -> None:
        result = handle_message("session-low-confidence", "Can you explain your wholesale partner rebate schedule?")
        self.assertIn("not confident enough", result["reply"].lower())
        self.assertEqual(result["tool_events"][-1]["name"], "create_ticket")


if __name__ == "__main__":
    unittest.main()
