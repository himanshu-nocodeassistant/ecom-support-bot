"""9d: Synthetic query generation.

Usage:
    python -m backend.eval.generate_queries --api-key $ANTHROPIC_API_KEY
    python -m backend.eval.generate_queries --api-key $ANTHROPIC_API_KEY --output backend/eval/queries_synthetic.json

Generates 5 paraphrased + 2 adversarial queries per KB document using Claude.
Near-duplicates (cosine similarity >= 0.92) are dropped via embedding comparison.
Output: backend/eval/queries_synthetic.json
The curated gold set (queries.json) is never modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).parent
DEFAULT_OUTPUT = EVAL_DIR / "queries_synthetic.json"

_GENERATION_PROMPT = """\
You are generating evaluation queries for a customer-support retrieval system.

Document title: {title}
Document content:
{content}

Generate exactly 7 queries a user might ask that would be answered by this document:
- 5 paraphrased queries (varied wording, same intent as the document content)
- 2 adversarial queries (ambiguous, misleading, or designed to confuse the retriever)

Respond with ONLY a JSON array. Each element must have:
  "query": the query string
  "type": one of "paraphrase" or "adversarial"

Example format:
[
  {{"query": "How do I get a refund?", "type": "paraphrase"}},
  {{"query": "Give me all my money back right now", "type": "adversarial"}}
]
"""


def generate_queries_for_doc(doc: dict[str, Any], client: Any) -> list[dict[str, Any]]:
    """Ask Claude to generate paraphrase + adversarial queries for one KB document.

    Returns a list of dicts with keys: query, type, document_id.
    Returns [] on any error.
    """
    try:
        prompt = _GENERATION_PROMPT.format(
            title=doc.get("title", ""),
            content=doc.get("content", ""),
        )
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        parsed = json.loads(text)
        result = []
        for item in parsed:
            query = item.get("query", "").strip()
            if not query:
                continue
            qtype = item.get("type", "paraphrase")
            result.append({"query": query, "type": qtype, "document_id": doc["id"]})
        return result
    except Exception:
        return []


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def deduplicate_by_embedding(
    queries: list[dict[str, Any]], threshold: float = 0.92
) -> list[dict[str, Any]]:
    """Drop near-duplicate queries using cosine similarity on their embedding vectors.

    Expects each item to have an "embedding" key. The embedding field is stripped
    from output items. Items without an embedding field are always kept.
    """
    kept: list[dict[str, Any]] = []
    kept_embeddings: list[list[float]] = []

    for item in queries:
        emb = item.get("embedding")
        if emb is None:
            out = {k: v for k, v in item.items() if k != "embedding"}
            kept.append(out)
            continue
        is_dup = any(_cosine(emb, existing) >= threshold for existing in kept_embeddings)
        if not is_dup:
            kept_embeddings.append(emb)
            out = {k: v for k, v in item.items() if k != "embedding"}
            kept.append(out)

    return kept


def generate_all(
    api_key: str,
    output_path: Path = DEFAULT_OUTPUT,
    voyage_api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Generate synthetic queries for all KB documents and write to output_path."""
    import anthropic

    from backend.app.data import KNOWLEDGE_BASE

    client = anthropic.Anthropic(api_key=api_key)
    all_queries: list[dict[str, Any]] = []

    print(f"Generating queries for {len(KNOWLEDGE_BASE)} documents...")
    for doc in KNOWLEDGE_BASE:
        print(f"  {doc['id']} — {doc['title']}")
        queries = generate_queries_for_doc(doc, client)
        all_queries.extend(queries)
        print(f"    {len(queries)} queries generated")

    # Deduplicate via embedding if Voyage key is available
    if voyage_api_key and all_queries:
        try:
            from backend.app.data_loader import embed_queries

            texts = [q["query"] for q in all_queries]
            embeddings = embed_queries(texts, api_key=voyage_api_key)
            for item, emb in zip(all_queries, embeddings):
                item["embedding"] = emb
            before = len(all_queries)
            all_queries = deduplicate_by_embedding(all_queries, threshold=0.92)
            print(f"Deduplication: {before} → {len(all_queries)} queries")
        except Exception as e:
            print(f"WARNING: embedding dedup skipped ({e})", file=sys.stderr)
            # Strip any partial embedding fields
            all_queries = [{k: v for k, v in q.items() if k != "embedding"} for q in all_queries]
    else:
        all_queries = deduplicate_by_embedding(all_queries)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_queries, indent=2))
    print(f"\n{len(all_queries)} synthetic queries written to {output_path}")
    return all_queries


def main() -> None:
    parser = argparse.ArgumentParser(description="9d: Synthetic query generator")
    parser.add_argument("--api-key", required=True, help="Anthropic API key")
    parser.add_argument(
        "--voyage-api-key", default=None, help="Voyage API key for dedup embeddings"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    generate_all(api_key=args.api_key, output_path=args.output, voyage_api_key=args.voyage_api_key)


if __name__ == "__main__":
    main()
