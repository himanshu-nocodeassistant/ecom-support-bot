"""Phase 8c — Extract durable memory facts from a completed conversation.

Called as a FastAPI BackgroundTask after a session ends.
Returns a list of {fact_type, fact_text, confidence} dicts ready to persist
via CustomerStore.save_memory_fact().
"""

from __future__ import annotations

import json
from typing import Any

_MIN_TURNS = 3
_CONFIDENCE_THRESHOLD = 0.7

_FACT_TYPES = ("order_preference", "issue_history", "product_interest", "communication_style")

_EXTRACTION_PROMPT = f"""You are analysing a customer support conversation to extract durable facts about the customer.

Extract up to 3 facts that would genuinely help a future agent serve this customer better.
Only include facts you are confident about (confidence ≥ {_CONFIDENCE_THRESHOLD}).

Valid fact_type values: {", ".join(_FACT_TYPES)}

Respond with a JSON array only — no other text. Example:
[
  {{"fact_type": "order_preference", "fact_text": "Prefers express shipping", "confidence": 0.9}},
  {{"fact_type": "issue_history", "fact_text": "Had a damaged item in order ORD-1001", "confidence": 0.85}}
]

If there are no facts worth keeping, respond with an empty array: []"""


def extract_facts_from_conversation(
    conversation: list[dict[str, Any]],
    api_key: str | None,
) -> list[dict[str, Any]]:
    """Return memory facts extracted from a completed conversation.

    Skips the API call for very short conversations or when no key is set.
    Returns [] on parse errors or empty conversations.
    """
    if not api_key:
        return []

    if len(conversation) < _MIN_TURNS:
        return []

    import anthropic

    transcript = "\n".join(f"{turn['role'].upper()}: {turn['content']}" for turn in conversation)

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=_EXTRACTION_PROMPT,
            messages=[{"role": "user", "content": transcript}],
        )
        raw = next((block.text for block in response.content if hasattr(block, "text")), "[]")
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        facts = json.loads(raw)
    except Exception:
        return []

    return [
        f
        for f in facts
        if isinstance(f, dict)
        and f.get("confidence", 0) >= _CONFIDENCE_THRESHOLD
        and "fact_type" in f
        and "fact_text" in f
    ]
