"""Small, optional, fail-open tracing boundary.

The application only depends on this module.  Langfuse is loaded lazily, so
offline tests and deployments without tracing credentials do not need it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol

from .config import get_settings

_LOG = logging.getLogger(__name__)
_ACTIVE: dict[str, list[Trace]] = {}


def safe_session_id(value: str) -> str:
    """Return a stable pseudonym. Never send the supplied value to a vendor."""
    salt = os.getenv("SUPPORTBOT_TRACE_SALT", "supportbot")
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:32]


def safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep trace fields operational; remove direct identifiers and free text."""
    direct = re.compile(
        r"email|order[_ -]?id|customer[_ -]?id|phone|address|query|input|output|content", re.I
    )
    result: dict[str, Any] = {}
    for key, value in metadata.items():
        if direct.search(str(key)):
            # Document IDs and ranks are explicitly useful and safe.
            if key not in {"document_ids", "candidate_document_ids", "order_id_hash"}:
                continue
        if isinstance(value, str | int | float | bool) or value is None:
            result[key] = value
        elif isinstance(value, list):
            clean_values = []
            for item in value:
                if isinstance(item, str | int | float | bool):
                    clean_values.append(item)
                elif isinstance(item, dict):
                    clean_values.append(
                        {
                            k: v
                            for k, v in item.items()
                            if k in {"id", "sparse", "dense", "fused", "rank"}
                        }
                    )
            result[key] = clean_values
    return result


class Trace(Protocol):
    def observe(self, name: str, metadata: dict[str, Any]) -> None: ...


class TracingBoundary(Protocol):
    @contextmanager
    def request(self, session_id: str, metadata: dict[str, Any]) -> Iterator[Trace]: ...


class _NoopTrace:
    def observe(self, name: str, metadata: dict[str, Any]) -> None:
        return None


class _FailOpenTrace:
    def __init__(self, inner: Trace) -> None:
        self.inner = inner

    def observe(self, name: str, metadata: dict[str, Any]) -> None:
        try:
            self.inner.observe(name, metadata)
        except Exception:
            _LOG.debug("Langfuse observation failed", exc_info=True)


class _NoopBoundary:
    @contextmanager
    def request(self, session_id: str, metadata: dict[str, Any]) -> Iterator[Trace]:
        yield _NoopTrace()


class FakeTracingBoundary:
    """Test boundary. It intentionally stores only the same safe fields."""

    def __init__(self) -> None:
        self.traces: list[dict[str, Any]] = []

    @contextmanager
    def request(self, session_id: str, metadata: dict[str, Any]) -> Iterator[Trace]:
        trace = {
            "id": safe_session_id(session_id),
            "metadata": safe_metadata(metadata),
            "observations": [],
            "closed": False,
        }
        self.traces.append(trace)

        class _FakeTrace:
            def observe(inner, name: str, fields: dict[str, Any]) -> None:
                trace["observations"].append({"name": name, "metadata": safe_metadata(fields)})

        try:
            yield _FakeTrace()
        finally:
            trace["closed"] = True


class LangfuseBoundary:
    def __init__(self, client: Any) -> None:
        self.client = client

    @contextmanager
    def request(self, session_id: str, metadata: dict[str, Any]) -> Iterator[Trace]:
        trace_id = safe_session_id(session_id)
        context = self.client.start_as_current_observation(
            name="chat_request",
            as_type="span",
            trace_context={"trace_id": trace_id},
            metadata=safe_metadata(metadata),
        )
        observation = context.__enter__()

        class _LangfuseTrace:
            def observe(inner, name: str, fields: dict[str, Any]) -> None:
                child = observation.start_observation(
                    name=name,
                    as_type="generation" if name == "generation" else "span",
                    metadata=safe_metadata(fields),
                )
                child.end()

        try:
            yield _LangfuseTrace()
        finally:
            try:
                context.__exit__(None, None, None)
                self.client.flush()
            except Exception:
                _LOG.debug("Langfuse flush failed", exc_info=True)


class _TraceHandle(_FailOpenTrace):
    def __init__(self, inner: Trace, context: Any) -> None:
        super().__init__(inner)
        self.context = context
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.context.__exit__(None, None, None)
        except Exception:
            _LOG.debug("Langfuse close failed", exc_info=True)


def get_tracing_boundary() -> TracingBoundary:
    settings = get_settings()
    if (
        not settings.trace_enabled
        or not settings.langfuse_public_key
        or not settings.langfuse_secret_key
    ):
        return _NoopBoundary()
    try:
        import langfuse

        client = langfuse.Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        return LangfuseBoundary(client)
    except Exception:
        _LOG.warning("Langfuse unavailable; continuing without tracing")
        return _NoopBoundary()


def timed_fields(start: float, **fields: Any) -> dict[str, Any]:
    return {**fields, "latency_ms": round((time.monotonic() - start) * 1000, 2)}


def close_trace(trace: Trace) -> None:
    try:
        close = getattr(trace, "close", None)
        if close:
            close()
    except Exception:
        _LOG.debug("Tracing close failed", exc_info=True)


def finish_trace(session_id: str) -> None:
    traces = _ACTIVE.get(session_id, [])
    if traces:
        trace = traces.pop()
        close_trace(trace)
    if not traces:
        _ACTIVE.pop(session_id, None)


def begin_trace(session_id: str, metadata: dict[str, Any]) -> Trace:
    """Start a trace without allowing setup or teardown to affect a request."""
    try:
        cm = get_tracing_boundary().request(session_id, metadata)
        handle = _TraceHandle(cm.__enter__(), cm)
        _ACTIVE.setdefault(session_id, []).append(handle)
        return handle
    except Exception:
        return _NoopTrace()
