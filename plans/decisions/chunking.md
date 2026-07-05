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

## Outcome (re-run 2026-07-04, doc-id hit-rate)

The original outcome below (avg Precision@3, +29% for semantic) was computed with
`_precision_at_k`, a title-string-matching metric later proven biased in favor of
finer chunking: more, smaller same-doc chunks can fill more top-3 slots than fewer,
larger ones, independent of retrieval quality (see `retrieval-finding.md`; §6g of the
2026-07-03 repurposing handoff). The audit has been re-run on `hit_rate_at_3`
(doc-id based, binary: did the expected document appear anywhere in the top-3),
which does not have this bias.

**Winner: fixed** — avg hit_rate@3 = 0.346 vs semantic = 0.327

| Strategy | avg hit_rate@3 | p50 latency | chunks |
|----------|----------------|-------------|--------|
| fixed    | 0.346          | 1.864s      | 58     |
| semantic | 0.327          | 1.922s      | 75     |

This reverses the original semantic-wins call. The gap is small (0.346 vs 0.327 on 24
queries — a couple of queries either way would flip it), so treat this as "no strong
evidence either strategy is better," not as a confident win for fixed. The active
configuration has not been changed pending a larger query set. **The +29%
semantic-wins claim must not be cited anywhere** (README, CV, etc.) going forward —
see the §8 claims ledger in the handoff doc.

---

### Original outcome (superseded — biased metric, do not cite)

**Winner: semantic** — avg Precision@3 = 0.130 vs fixed = 0.101 (+29% relative)

| Strategy | avg P@3 | p50 latency | chunks |
|----------|---------|-------------|--------|
| fixed    | 0.101   | 2.175s      | 6      |
| semantic | 0.130   | 2.117s      | 12     |

Semantic chunking produces more, finer-grained chunks (12 vs 6) that map more tightly to individual facts. The retriever fetches the exact relevant sentence rather than a merged block mixing unrelated facts. Latency is slightly better because smaller chunks embed faster.

The knowledge base was re-imported with `strategy="semantic"` as the active configuration at the time; this decision is superseded by the re-run above.
