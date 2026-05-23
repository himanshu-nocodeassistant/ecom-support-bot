# Chunking Strategy Decision

## Current approach (fixed)

- **Strategy**: paragraph accumulation up to `target_chunk_size=220` chars
- **Splitter**: `"\n"` line boundaries
- **Overlap**: none
- **Location**: `backend/app/data_loader._chunk_fixed`

Paragraphs are greedily merged into one chunk until adding the next paragraph would exceed 220 characters. This means a single chunk may contain 2–3 short paragraphs from different subtopics within the same document.

## Alternative (semantic)

- **Strategy**: one paragraph → one chunk, no cross-paragraph merging
- **Splitter**: `"\n"` line boundaries; each paragraph is its own chunk
- **Overlap**: none
- **Location**: `backend/app/data_loader._chunk_semantic`

Preserves natural topic boundaries. Each paragraph covers a single fact (e.g. "Battery life is up to 30 hours") — keeping it isolated means the retriever fetches exactly the relevant sentence rather than a merged block containing unrelated facts.

## How to reproduce the benchmark

```bash
python -m backend.eval.run --chunking-audit --knowledge-dir backend/knowledge
```

Results are written to `backend/eval/results/chunking_audit.json`.

## Outcome

**Winner: semantic** — avg Precision@3 = 0.130 vs fixed = 0.101 (+29% relative)

| Strategy | avg P@3 | p50 latency | chunks |
|----------|---------|-------------|--------|
| fixed    | 0.101   | 2.175s      | 6      |
| semantic | 0.130   | 2.117s      | 12     |

Semantic chunking produces more, finer-grained chunks (12 vs 6) that map more tightly to individual facts. The retriever fetches the exact relevant sentence rather than a merged block mixing unrelated facts. Latency is slightly better because smaller chunks embed faster.

The knowledge base has been re-imported with `strategy="semantic"` as the active configuration.
