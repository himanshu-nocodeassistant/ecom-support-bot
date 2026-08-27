from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from backend.app.tracing import (
    FakeTracingBoundary,
    LangfuseBoundary,
    begin_trace,
    close_trace,
    safe_session_id,
)


def test_session_correlation_is_pseudonymous_and_not_direct_identifier() -> None:
    value = safe_session_id("alice@example.com/ORD-1001")
    assert value != "alice@example.com/ORD-1001"
    assert "alice@example.com" not in value
    assert "ORD-1001" not in value
    assert value == safe_session_id("alice@example.com/ORD-1001")


def test_fake_boundary_records_safe_observations() -> None:
    boundary = FakeTracingBoundary()
    with boundary.request("session-1", {"mode": "phase3"}) as trace:
        trace.observe("sparse_retrieval", {"candidate_depth": 20, "document_ids": ["doc-1"]})

    assert boundary.traces[0]["metadata"]["mode"] == "phase3"
    assert boundary.traces[0]["observations"][0]["name"] == "sparse_retrieval"
    assert boundary.traces[0]["observations"][0]["metadata"]["document_ids"] == ["doc-1"]


def test_tracing_failure_is_fail_open() -> None:
    class BrokenBoundary:
        def request(self, *_args, **_kwargs):
            raise RuntimeError("langfuse unavailable")

    with patch("backend.app.tracing.get_tracing_boundary", return_value=BrokenBoundary()):
        from backend.app.agent import _handle_message_deterministic

        result = _handle_message_deterministic("s1", "hello", mode="phase3")
    assert result["mode"] == "phase3"


def test_trace_close_runs_after_success_and_exception() -> None:
    boundary = FakeTracingBoundary()
    with patch("backend.app.tracing.get_tracing_boundary", return_value=boundary):
        trace = begin_trace("success", {"mode": "phase3"})
        trace.observe("generation", {"model": "test"})
        close_trace(trace)
        assert boundary.traces[-1]["closed"] is True

        trace = begin_trace("failure", {"mode": "phase3"})
        try:
            raise ValueError("provider failed")
        except ValueError:
            close_trace(trace)
        assert boundary.traces[-1]["closed"] is True


def test_langfuse_v3_client_contract_is_used_and_metadata_is_safe() -> None:
    class Observation:
        def __init__(self, calls):
            self.calls = calls

        def start_observation(self, **kwargs):
            self.calls.append(("child", kwargs))
            return self

        def end(self):
            self.calls.append(("end", {}))

    class Client:
        def __init__(self):
            self.calls = []
            self.observation = Observation(self.calls)

        def start_as_current_observation(self, **kwargs):
            self.calls.append(("root", kwargs))

            class Context:
                def __enter__(inner):
                    return self.observation

                def __exit__(inner, *_args):
                    self.calls.append(("close", {}))

            return Context()

        def flush(self):
            self.calls.append(("flush", {}))

    client = Client()
    boundary = LangfuseBoundary(client)
    with boundary.request(
        "alice@example.com/ORD-1001", {"query": "secret", "mode": "phase3"}
    ) as trace:
        trace.observe("tool_call", {"name": "lookup_order", "order_id": "ORD-1001"})
    root = next(call for name, call in client.calls if name == "root")
    assert len(root["trace_context"]["trace_id"]) == 32
    assert "query" not in root["metadata"]
    assert (
        "order_id" not in next(call for name, call in client.calls if name == "child")["metadata"]
    )
    assert any(name == "flush" for name, _ in client.calls)


def test_chat_emits_one_closed_trace_for_retrieval_tool_and_generation() -> None:
    boundary = FakeTracingBoundary()

    class Repo:
        tracer = None

        def search_knowledge(self, query, k=None):
            self.tracer.observe(
                "sparse_retrieval",
                {
                    "candidate_depth": 2,
                    "final_depth": 1,
                    "document_ids": ["doc-returns"],
                    "stage_ranks": [1],
                    "degraded": False,
                },
            )
            self.tracer.observe(
                "reranking",
                {
                    "candidate_depth": 2,
                    "final_depth": 1,
                    "document_ids": ["doc-returns"],
                    "degraded": False,
                },
            )
            return [
                {
                    "id": "doc-returns",
                    "title": "Returns",
                    "category": "policy",
                    "content": "Returns are accepted.",
                    "score": 0.9,
                }
            ]

        def get_order(self, order_id):
            return None

    tool_block = SimpleNamespace(
        type="tool_use",
        id="tool-1",
        name="search_knowledge_base",
        input={"query": "return my order"},
    )
    text_block = SimpleNamespace(type="text", text="Returns are accepted.")
    responses = [
        SimpleNamespace(
            content=[tool_block],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=10, output_tokens=2),
        ),
        SimpleNamespace(
            content=[text_block],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=20, output_tokens=5),
        ),
    ]
    fake_client = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: responses.pop(0)))
    settings = SimpleNamespace(anthropic_api_key="test", data_backend="memory")
    with (
        patch("backend.app.tracing.get_tracing_boundary", return_value=boundary),
        patch("backend.app.agent.get_settings", return_value=settings),
        patch("backend.app.agent._repo_for_mode", return_value=Repo()),
        patch("anthropic.Anthropic", return_value=fake_client),
    ):
        from backend.app.agent import handle_message

        result = handle_message(
            "alice@example.com/ORD-1001", "return my order", customer_email="alice@example.com"
        )

    assert result["reply"] == "Returns are accepted."
    assert len(boundary.traces) == 1
    trace = boundary.traces[0]
    assert trace["closed"] is True
    names = {item["name"] for item in trace["observations"]}
    assert {"sparse_retrieval", "reranking", "tool_call", "generation"} <= names
    serialized = repr(trace)
    assert "alice@example.com" not in serialized
    assert "ORD-1001" not in serialized
    assert "return my order" not in serialized


def test_tracing_can_be_disabled_without_credentials() -> None:
    settings = SimpleNamespace(
        trace_enabled=False,
        langfuse_public_key="key",
        langfuse_secret_key="secret",
        langfuse_host="https://example.test",
    )
    with patch("backend.app.tracing.get_settings", return_value=settings):
        from backend.app.tracing import get_tracing_boundary

        assert get_tracing_boundary().__class__.__name__ == "_NoopBoundary"
