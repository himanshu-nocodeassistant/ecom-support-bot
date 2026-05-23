# Release Roadmap

Each release adds one meaningful capability, keeps the system runnable, and ships with a short note about what improved and what remains weak.

---

## `v0.1.0-core` ✅ Complete

Ship a working baseline:

- Minimal support API
- Keyword retrieval over a small knowledge base
- Order lookup tool with Supabase + Olist dataset
- Refund and ticket stubs
- Session memory
- Repository layer (in-memory and Postgres-backed)
- Basic tests

What was noted:

- Retrieval quality is shallow — keyword count only, semantic gaps are a known miss
- Tool routing is rule-based (if/elif phrase matching), not model-driven
- No streaming, no observability
- Knowledge retrieval uses persisted Supabase chunks, but only full-text search

---

## `v0.2.0-retrieval-upgrade` ✅ Complete

Improved answer quality:

- Document chunking with metadata preservation
- Voyage AI `voyage-3-lite` embeddings (512 dims) on all knowledge chunks
- HNSW index on `knowledge_chunks.embedding` for fast similarity search
- Hybrid search: `score = 0.3 × ts_rank + 0.7 × (1 − cosine_distance)`
- Confidence threshold raised from 0.05 → 0.25 to match continuous float scores
- `compare-retrieval` CLI command for side-by-side FTS vs hybrid comparison
- Fallback chain: hybrid → full-text → in-memory keyword

What improved:

- "Can I get my money back?" → refund policy (0.266) — previously escalated
- "How long until my package shows up?" → shipping policy (0.273) — previously escalated
- "My purchase arrived damaged, I want compensation" → refund policy (0.329) — previously escalated
- "Does the portable blender have a safety lock?" → score 0.215 (FTS) vs 0.536 (hybrid) — both correct, hybrid more confident

What remains weak:

- Routing is still deterministic; LLM not yet in the loop
- Replies are template strings, not model-generated
- 0.25 threshold is a heuristic

---

## `v0.3.0-agent-tool-loop` ✅ Complete

Made the assistant genuinely agentic:

- Claude API tool loop replacing if/elif routing
- Four formal JSON tool schemas: `lookup_order`, `request_refund`, `search_knowledge_base`, `create_ticket`
- Multi-step tool chains in a single user message (e.g. lookup → refund)
- Full conversation history passed to model each turn — no more regex-based order ID recall
- Contextual, model-generated ticket subjects and descriptions
- Graceful degradation: deterministic fallback when no `ANTHROPIC_API_KEY` is set

What works now that didn't before:

- "My order arrived damaged, I want a refund" → `lookup_order` + `request_refund` in one turn
- "Can I get my money back?" → bot asks for order ID, user provides it, refund completes — two natural turns
- Refund for undelivered order → blocked with alternatives offered, not silent failure
- "When will my order arrive?" three turns later (after topic switch) → answered from history, no re-lookup
- Out-of-scope inquiry → conversational clarification before ticket, descriptive ticket subject

What remains weak:

- No streaming — full loop completes before user sees any output
- No UI — raw JSON API only
- Session memory is in-process (lost on server restart)

---

## `v0.4.0-streaming-ui` ✅ Complete

Improved the product feel:

- `POST /api/chat/stream` — SSE streaming with `tool_start`, `tool_result`, `token`, `done`, `error` events
- `POST /api/compare` — runs phase1, phase2, phase3 concurrently and returns all three in one call
- `frontend/index.html` — single-file Chat UI (streaming) + Compare tab (three-column phase comparison)
- `mode` param on every request — switch phase without restarting the server
- CORS middleware; 17 new tests covering all modes and SSE format

What improved:

- Tool activity visible as it executes, not only in the final JSON response
- Compare tab makes the cumulative improvement across phases visible in one query
- Project feels like a product, not a raw API

What remains weak:

- Session memory still in-process — lost on server restart
- No evaluation layer — improvements demonstrated by example, not measured systematically

---

## `v0.5.0-evaluation` ✅ Complete

Made retrieval quality measurable and every upgrade justifiable with numbers:

**5a — Ground truth dataset**
- `backend/eval/queries.json`: 30 labeled queries across 7 categories

**5b — Chunking strategy audit**
- `"fixed"` (220-char accumulation) vs `"semantic"` (one paragraph per chunk)
- `python -m backend.eval.run --chunking-audit` benchmarks both, writes to `backend/eval/results/`
- Decision recorded in `plans/decisions/chunking.md`

**5c — Metadata enrichment**
- `doc_type`, `source_section`, `chunk_strategy` in every chunk JSONB metadata
- `date_updated` on `knowledge_documents`; GIN index on `metadata->>'category'`
- Migration: `backend/sql/migrate_5c_metadata.sql`

**5d — Deduplication**
- `deduplicate_chunks()` in `data_loader.py` drops near-duplicates (cosine ≥ 0.95) at index time
- Count logged and returned in import result

**5e — Advanced retrieval**
- Voyage `rerank-2-lite` reranking as post-retrieval step (`enable_reranking` flag)
- Keyword-based category pre-filtering (`enable_metadata_filter` flag)
- Both off by default, toggled per eval mode

**5f — Citation grounding**
- `sources` array on every API response and SSE `done` event; scores < 0.15 suppressed
- Source chips rendered below each bot reply in the UI

**5g/5h — Full metrics + A/B runner**
- Precision@3, Recall@3, context relevance, latency p50/p95, estimated Voyage cost
- `python -m backend.eval.run --all-modes` runs all 5 modes and prints comparison table
- Per-mode JSON in `backend/eval/results/`

**5i — Eval dashboard**
- `GET /eval` on FastAPI — color-coded summary table, latency bar chart, per-query drill-down

What remains weak:

- Session memory still in-process
- Answer correctness is keyword-overlap only; LLM-judge scoring deferred
- Eval covers retrieval quality, not end-to-end agent quality

---

## `v0.6.0-data-benchmark` — Planned

Make every improvement data-backed: committed benchmark numbers, LLM-judge correctness scoring, agent quality fixtures, and a CI regression gate. The goal is a repo where readers see actual metric deltas, not prose claims.

**6a — Baseline benchmark run**
- Run `--all-modes` against live Supabase+Voyage stack; commit raw JSON results
- Generate `docs/benchmark.md` — a markdown comparison table readers can inspect
- This becomes the "before" every future change is measured against

**6b — LLM-judge answer correctness**
- Replace keyword-overlap correctness with a Claude API call scoring factual accuracy (0–1)
- Adds a meaningful correctness column to the benchmark table
- Flags hallucinations and wrong answers that keyword overlap misses

**6c — Agent quality fixtures**
- 10–15 scripted multi-turn conversations with expected tool sequences
- Metrics: tool selection accuracy, unnecessary tool calls, correct refusals on unanswerable queries
- Closes the gap: retrieval quality ≠ agent quality

**6d — Regression gate**
- GitHub Actions workflow that runs eval on every PR
- Fails if any metric drops more than a defined threshold from the committed baseline
- Turns the benchmark from a one-time snapshot into a living guard
