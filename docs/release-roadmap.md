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

## `v0.6.0-data-benchmark` ✅ Complete

Made every improvement data-backed: committed benchmark numbers, LLM-judge correctness scoring, agent quality fixtures, and a CI regression gate.

**6a — Baseline benchmark run**
- `--benchmark` flag writes `docs/benchmark.md` — mode × metric table committed to repo
- `backend/eval/results/baseline.json` — snapshot for regression comparisons
- `python -m backend.eval.run --all-modes --benchmark` regenerates everything

**6b — LLM-judge answer correctness**
- `--llm-judge` flag calls `claude-haiku-4-5-20251001` per query; scores 0–1 factual accuracy
- `answer_correctness_llm` column alongside existing `answer_correctness_kw`
- Per-query judge cost tracked in results JSON and benchmark table

**6c — Agent quality fixtures**
- `backend/eval/agent_fixtures.json` — 12 multi-turn fixtures
- Metrics: tool selection accuracy, extra tool calls, refusal correctness
- `python -m backend.eval.run --agent-eval`

**6d — Regression gate**
- `.github/workflows/eval.yml` — runs on every PR to main
- `backend/eval/check_regression.py` — exits 1 on >10% drop; posts delta as PR comment
- `backend/eval/thresholds.json` — configurable per-metric tolerances

---

## `v0.7.0-persistent-customer-memory` 🟡 Data layer complete

Customer identity, conversation persistence, and memory facts are designed and tested in-memory. Nothing is wired into the live agent yet.

What exists:

- `InMemoryCustomerStore` — identity upsert, session linking, memory facts (confidence-gated), order linkage (sliding window of 5)
- `InMemoryConversationStore` — turn persistence, max-window load
- `build_customer_context()` / `build_system_prompt()` helpers
- `backend/sql/migrate_7_customer_memory.sql` — 5 new tables
- 31 passing tests in `test_customer_memory.py`

What does not exist yet:

- Nothing called from `agent.py`; `SESSION_MEMORY` dict still in use
- Fact extraction never runs
- `customer_email` not accepted by the API
- No Postgres backend for either store
