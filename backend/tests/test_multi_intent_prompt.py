"""Prompt-level contracts for safe multi-intent support handling.

These are deliberately deterministic: live-model behaviour is measured by the
adversarial suite, while this test prevents the routing rules that guide that
behaviour from silently disappearing in a prompt edit.
"""

from backend.app.prompts import SYSTEM_PROMPT


def test_prompt_requires_supported_intents_to_be_handled_together() -> None:
    assert "Handle every supported intent in a multi-part request" in SYSTEM_PROMPT
    assert "look up an identified" in SYSTEM_PROMPT
    assert "order and search the knowledge base" in SYSTEM_PROMPT


def test_prompt_requires_safe_handling_of_unsupported_intents() -> None:
    assert "Do not claim these" in SYSTEM_PROMPT
    assert "actions happened" in SYSTEM_PROMPT
    assert "Create a ticket for each unsupported part" in SYSTEM_PROMPT


def test_prompt_blocks_refunds_without_a_customer_reason() -> None:
    assert (
        "Never call request_refund until the customer has supplied a concrete refund reason"
        in SYSTEM_PROMPT
    )
    assert "ask for the reason instead of processing it" in SYSTEM_PROMPT


def test_prompt_requires_exact_knowledge_base_citations() -> None:
    assert "[source:<document-id>]" in SYSTEM_PROMPT
    assert "Do not cite documents that do not support the claim" in SYSTEM_PROMPT
