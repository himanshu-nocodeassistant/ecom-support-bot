"""Phase 7 — Customer context building for agent system prompt injection."""

from __future__ import annotations

from .agent import SYSTEM_PROMPT


def build_customer_context(
    facts: list[dict],
    prior_order_ids: list[str],
) -> str:
    """Return a markdown block summarising what we know about this customer.

    Returns an empty string when there is nothing to surface (anonymous session
    or first-time customer with no history).
    """
    lines: list[str] = []

    if prior_order_ids:
        ids = ", ".join(prior_order_ids)
        lines.append(f"Prior orders on file: {ids}")

    for fact in facts:
        lines.append(fact["fact_text"])

    if not lines:
        return ""

    body = "\n".join(f"- {line}" for line in lines)
    return f"## Customer context\n{body}"


def build_system_prompt(customer_context: str = "") -> str:
    """Return the full system prompt, optionally prefixed with customer context."""
    if not customer_context:
        return SYSTEM_PROMPT
    return f"{customer_context}\n\n{SYSTEM_PROMPT}"
