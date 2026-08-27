from pathlib import Path

from backend.eval.run import evaluate_end_to_end, extract_citations, send_experiment_scores


def test_e2e_scores_supported_claims_and_multiple_expected_documents():
    cases = [
        {
            "id": "supported",
            "query": "delivery and returns",
            "reference_claims": [
                "Delivery takes 3 to 7 business days",
                "Returns are allowed within 30 days",
            ],
            "expected_supporting_document_ids": ["shipping-policy", "returns-policy"],
            "answerable": True,
        }
    ]
    docs = [
        {"id": "shipping-policy", "content": "Delivery takes 3 to 7 business days."},
        {"id": "returns-policy", "content": "Returns are allowed within 30 days."},
    ]
    result = evaluate_end_to_end(
        cases,
        retrieve=lambda _: docs,
        generate=lambda *_: {
            "reply": "Delivery takes 3 to 7 business days [source:shipping-policy]. Returns are allowed within 30 days [source:returns-policy].",
            "sources": docs,
        },
    )
    assert result["avg_faithfulness"] == 1.0
    assert result["avg_citation_accuracy"] == 1.0
    assert result["avg_correct_refusal"] == 1.0


def test_e2e_detects_unsupported_answer_and_incorrect_citation():
    case = {
        "id": "bad",
        "query": "refund",
        "reference_claims": ["Refunds require delivery"],
        "expected_supporting_document_ids": ["refund-policy"],
        "answerable": True,
    }
    result = evaluate_end_to_end(
        [case],
        retrieve=lambda _: [{"id": "refund-policy", "content": "Refunds require delivery."}],
        generate=lambda *_: {
            "reply": "Refunds are instant [source:shipping-policy].",
            "sources": [],
        },
    )
    assert result["avg_faithfulness"] == 0.0
    assert result["avg_citation_accuracy"] == 0.0


def test_unsupported_question_requires_refusal_or_escalation():
    case = {
        "id": "oos",
        "query": "weather",
        "reference_claims": [],
        "expected_supporting_document_ids": [],
        "answerable": False,
    }
    refused = evaluate_end_to_end(
        [case],
        retrieve=lambda _: [],
        generate=lambda *_: {
            "reply": "I cannot answer that. I escalated this to a human.",
            "escalated": True,
        },
    )
    answered = evaluate_end_to_end(
        [case], retrieve=lambda _: [], generate=lambda *_: {"reply": "It is sunny today."}
    )
    assert refused["avg_correct_refusal"] == 1.0
    assert answered["avg_correct_refusal"] == 0.0


def test_citations_are_extracted_from_structured_sources_and_text():
    assert extract_citations("See [source:a] and [citation: b].") == {"a", "b"}


def test_chat_path_uses_search_tool_evidence_with_content():
    case = {
        "id": "tool",
        "query": "q",
        "reference_claims": ["A fact"],
        "expected_supporting_document_ids": ["doc"],
        "answerable": True,
    }
    result = evaluate_end_to_end(
        [case],
        chat=lambda *_: {
            "reply": "A fact [source:doc]",
            "sources": [],
            "tool_events": [
                {
                    "name": "search_knowledge_base",
                    "output": {"matches": [{"id": "doc", "content": "A fact"}]},
                }
            ],
        },
    )
    assert result["avg_faithfulness"] == 1.0


def test_extra_and_unsupported_citations_are_penalized():
    case = {
        "id": "cite",
        "query": "q",
        "reference_claims": ["A fact"],
        "expected_supporting_document_ids": ["doc"],
        "answerable": True,
    }
    result = evaluate_end_to_end(
        [case],
        retrieve=lambda _: [{"id": "doc", "content": "A fact"}],
        generate=lambda *_: {"reply": "A fact [source:doc] [source:wrong]"},
    )
    assert result["avg_citation_accuracy"] == 0.0


def test_langfuse_v3_uses_create_score():
    class FakeClient:
        def __init__(self):
            self.calls = []

        def create_score(self, **kwargs):
            self.calls.append(kwargs)

    client = FakeClient()
    result = {"avg_faithfulness": 1.0, "avg_citation_accuracy": 0.5, "avg_correct_refusal": 1.0}
    assert send_experiment_scores(result, client)
    assert len(client.calls) == 3
    assert all(call["name"].startswith("e2e_") for call in client.calls)


def test_workflow_runs_e2e_before_strict_regression():
    workflow = Path(".github/workflows/eval.yml").read_text()
    assert workflow.index("--e2e-eval") < workflow.index("check_regression --strict")


def test_optional_judge_metadata_and_fail_open_sink():
    case = {
        "id": "x",
        "query": "q",
        "reference_claims": ["A fact"],
        "expected_supporting_document_ids": ["doc"],
        "answerable": True,
    }
    result = evaluate_end_to_end(
        [case],
        retrieve=lambda _: [{"id": "doc", "content": "A fact"}],
        generate=lambda *_: {"reply": "A fact [source:doc]"},
        judge=lambda *_: {"faithfulness": 1.0},
        judge_model="judge-v1",
        prompt_version="p2",
        dataset_version="d3",
        score_sink=lambda _: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    assert result["judge"]["model"] == "judge-v1"
    assert result["judge"]["prompt_version"] == "p2"
    assert result["judge"]["dataset_version"] == "d3"
    assert "runtime_s" in result["judge"]
