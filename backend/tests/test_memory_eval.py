"""Phase 8g — memory_recall_rate eval metric (TDD: red phase).

memory_recall_rate measures what fraction of memory fixtures surface the expected
context fragments in the agent's system prompt during a second-session call.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


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


class MemoryRecallRateTests(unittest.TestCase):
    """memory_recall_rate() computes the fraction of fixtures that recall correctly."""

    def test_importable(self) -> None:
        from backend.eval.memory_eval import memory_recall_rate  # noqa

    def test_perfect_recall_returns_one(self) -> None:
        from backend.eval.memory_eval import memory_recall_rate

        # All fragments present → 1.0
        rate = memory_recall_rate(fixtures_recalled=5, fixtures_total=5)
        self.assertAlmostEqual(rate, 1.0)

    def test_zero_recall_returns_zero(self) -> None:
        from backend.eval.memory_eval import memory_recall_rate

        rate = memory_recall_rate(fixtures_recalled=0, fixtures_total=5)
        self.assertAlmostEqual(rate, 0.0)

    def test_partial_recall(self) -> None:
        from backend.eval.memory_eval import memory_recall_rate

        rate = memory_recall_rate(fixtures_recalled=3, fixtures_total=5)
        self.assertAlmostEqual(rate, 0.6)

    def test_zero_total_returns_zero(self) -> None:
        from backend.eval.memory_eval import memory_recall_rate

        rate = memory_recall_rate(fixtures_recalled=0, fixtures_total=0)
        self.assertAlmostEqual(rate, 0.0)


class RunMemoryEvalTests(unittest.TestCase):
    """run_memory_eval() populates CustomerStore with fixtures and checks system prompts."""

    def setUp(self) -> None:
        self._settings_patcher = patch(
            "backend.app.agent.get_settings",
            return_value=MagicMock(anthropic_api_key="test-key"),
        )
        self._settings_patcher.start()
        self._repo_patcher = patch(
            "backend.app.agent.get_repository",
            return_value=MagicMock(),
        )
        self._repo_patcher.start()

    def tearDown(self) -> None:
        self._settings_patcher.stop()
        self._repo_patcher.stop()

    def _patch_anthropic(self, reply: str = "Got it.") -> MagicMock:
        client = MagicMock()
        client.messages.create.return_value = _response("end_turn", _text_block(reply))
        patcher = patch("anthropic.Anthropic", return_value=client)
        patcher.start()
        self.addCleanup(patcher.stop)
        return client

    def test_run_memory_eval_importable(self) -> None:
        from backend.eval.memory_eval import run_memory_eval  # noqa

    def test_run_memory_eval_returns_rate_and_count(self) -> None:
        from backend.eval.memory_eval import run_memory_eval

        fixtures = [
            {
                "fixture_id": "mem-001",
                "stored_facts": [
                    {
                        "fact_type": "order_preference",
                        "fact_text": "Prefers express shipping",
                        "confidence": 0.9,
                    }
                ],
                "prior_orders": [],
                "expected_context_fragments": ["Prefers express shipping"],
            }
        ]
        self._patch_anthropic()

        result = run_memory_eval(fixtures)

        self.assertIn("memory_recall_rate", result)
        self.assertIn("recalled", result)
        self.assertIn("total", result)
        self.assertGreaterEqual(result["memory_recall_rate"], 0.0)
        self.assertLessEqual(result["memory_recall_rate"], 1.0)

    def test_fixture_with_prior_orders_recalled(self) -> None:
        from backend.eval.memory_eval import run_memory_eval

        fixtures = [
            {
                "fixture_id": "mem-order",
                "stored_facts": [],
                "prior_orders": ["ORD-9999"],
                "expected_context_fragments": ["ORD-9999"],
            }
        ]
        self._patch_anthropic()

        result = run_memory_eval(fixtures)

        self.assertEqual(result["recalled"], 1)
        self.assertAlmostEqual(result["memory_recall_rate"], 1.0)

    def test_anonymous_fixture_passes_when_no_context_expected(self) -> None:
        from backend.eval.memory_eval import run_memory_eval

        fixtures = [
            {
                "fixture_id": "mem-anon",
                "stored_facts": [],
                "prior_orders": [],
                "expected_context_fragments": [],
            }
        ]
        self._patch_anthropic()

        result = run_memory_eval(fixtures)

        self.assertEqual(result["recalled"], 1)

    def test_low_confidence_facts_not_recalled(self) -> None:
        from backend.eval.memory_eval import run_memory_eval

        fixtures = [
            {
                "fixture_id": "mem-low-conf",
                "stored_facts": [
                    {
                        "fact_type": "order_preference",
                        "fact_text": "Maybe prefers fast shipping",
                        "confidence": 0.5,
                    }
                ],
                "prior_orders": [],
                "expected_context_fragments": [],
            }
        ]
        self._patch_anthropic()

        result = run_memory_eval(fixtures)

        # Low-confidence facts are never stored, so no context to surface — still "recalled" (no fragments expected)
        self.assertEqual(result["recalled"], 1)


class MemoryRegressionGateTests(unittest.TestCase):
    """The regression gate must enforce memory_recall_rate threshold."""

    def test_memory_metric_added_to_thresholds(self) -> None:
        from backend.eval.check_regression import check_memory_regression

        thresholds = {"memory_recall_rate_min": 0.75}
        baseline = {"memory_recall_rate": 0.90}
        current = {"memory_recall_rate": 0.90}

        failures = check_memory_regression(thresholds, baseline, current)
        self.assertEqual(failures, [])

    def test_memory_regression_detected(self) -> None:
        from backend.eval.check_regression import check_memory_regression

        thresholds = {"memory_recall_rate_min": 0.75}
        baseline = {"memory_recall_rate": 0.90}
        current = {"memory_recall_rate": 0.60}

        failures = check_memory_regression(thresholds, baseline, current)
        self.assertEqual(len(failures), 1)
        self.assertIn("memory_recall_rate", failures[0])

    def test_memory_rate_above_minimum_passes(self) -> None:
        from backend.eval.check_regression import check_memory_regression

        thresholds = {"memory_recall_rate_min": 0.75}
        baseline = {"memory_recall_rate": 0.80}
        current = {"memory_recall_rate": 0.80}

        failures = check_memory_regression(thresholds, baseline, current)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
