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

---

## `v0.8.0-memory-wiring` ✅ Complete

Wired the customer-memory data layer into the live agent.

- `SESSION_MEMORY` dict replaced by injected `ConversationStore` in both agent handlers
- `CustomerStore` wired in — upserts customer by email, links session, injects prior orders + facts into system prompt
- `lookup_order` tool results auto-link order IDs to the customer
- `fact_extractor.py` — Claude Haiku extracts 1–3 confidence-gated facts from completed conversations
- `PostgresConversationStore` and `PostgresCustomerStore` (psycopg2) with in-memory fallback
- 90-day TTL on `customer_memory` facts; SQL migration `migrate_8_memory_wiring.sql`
- `SYSTEM_PROMPT` extracted to `prompts.py` to break circular import
- `customer_email` field on `/api/chat` and `/api/chat/stream`
- SSE `done` event includes `returning_customer: bool`
- Frontend: email input row, welcome-back banner, `✓ linked` status
- `memory_eval.py` with `memory_recall_rate` metric; 7 multi-session fixtures
- `check_memory_regression()` in CI gate; `memory_recall_rate_min: 0.75` in `thresholds.json`
- `/eval/memory` endpoint and dashboard panel

What improved:

- Returning customers receive personalised responses without re-identifying themselves
- Prior order IDs surface in the system prompt automatically
- Facts about preferences and issue history accumulate across sessions and expire after 90 days

What remains weak:

- `customer_email` is taken at face value — no OTP or token verification
- Fact extraction fires on demand rather than as a true background task post-session
- Postgres stores selected only when `DATABASE_URL` is set; no automatic migration runner

---

## `v0.9.0-rag-honesty-and-polish` 🚧 Planned

Resolve the keyword-beats-hybrid finding from the Phase 6 baseline and bring eval coverage, README, and repo hygiene up to current state.

### 9a — Retrieval finding case study

The Phase 6 baseline shows in-memory keyword retrieval (P@3=0.290, R@3=0.870) outperforming every Postgres mode (best hybrid P@3=0.116). `thresholds.json` declares `hybrid+rerank+filter` as `best_mode` — which is the worst performer. The CI regression gate is guarding the wrong mode.

Two root-cause candidates:

- **Metric bias.** `_precision_at_k` compares chunk titles by string equality. Keyword returns whole docs; chunked modes return 3 chunks of the same doc. A correct hit from a chunked mode only scores 1/3 if the other two slots pull from different docs.
- **KB-size effect.** 4 docs / 6 chunks give retrieval modes almost nothing to separate. The 70/30 hybrid weight amplifies embedding noise; `_infer_category` pre-filter can reduce candidates to 1–2, often the wrong one.

Deliverables:

- `plans/decisions/retrieval-finding.md` — root-cause investigation, before/after numbers, final verdict
- Switch P@3/R@3 to `document_id`-based comparison; re-run all modes
- Commit fresh `comparison.json` and `docs/benchmark.md`
- Update `thresholds.json` `best_mode` to reflect the corrected ranking

### 9b — Knowledge base expansion

- Grow KB to ≥ 15 docs: returns vs refunds distinction, warranty, payment methods, account management, order modification, address change, gift wrapping, subscription billing, B2B/wholesale, accessibility
- Add 2–3 product guides for additional SKUs
- Re-run `--chunking-audit` after expansion
- Extend `backend/eval/queries.json` to ≥ 60 queries covering new docs

### 9c — Eval metric fixes

- Switch P@3/R@3 to `document_id`-based comparison (fixes structural bias against chunked modes)
- Add `hit_rate_at_k` for k ∈ {1, 3, 5, 10}
- Add NDCG@5 with per-query relevance ranking
- Add MRR (Mean Reciprocal Rank)
- Update `docs/benchmark.md` columns; deprecate (not delete) old title-based P@3 with an explanation

### 9d — Synthetic query generation

- `backend/eval/generate_queries.py` — ask Claude to produce 5 paraphrased + 2 adversarial queries per doc
- Dedupe via embedding cosine similarity (drop ≥ 0.92)
- Output to `backend/eval/queries_synthetic.json`; curated `queries.json` stays as gold set
- Eval runner gains `--query-set gold|synthetic|both`

### 9e — Adversarial eval set

- 10 prompt-injected queries
- 10 ambiguous queries that should trigger clarification
- 10 multi-intent queries that should trigger multiple tools
- 10 out-of-scope queries that should refuse cleanly
- New metrics: `injection_refusal_rate`, `clarification_rate`, `multi_tool_rate`, `oos_refusal_rate`
- CI gate at ≥ 0.80 for refusal metrics

### 9f — Cost-quality Pareto visualisation

- Pareto plot (cost on x, NDCG@5 on y, one dot per mode) as SVG embedded in `docs/benchmark.md`
- Latency-quality version (p95 on x, NDCG@5 on y)
- Regenerated automatically by `--benchmark` flag

### 9g — Benchmark trend history

- Append every CI eval run's summary metrics to `docs/benchmark-history.jsonl`
- "Trend" section in `docs/benchmark.md` with sparklines for the last 20 runs
- CI workflow commits the updated history file

### 9h — README + dashboard polish

- Refresh capabilities table to current state
- Rewrite intro paragraph to current project description
- Add screenshots: `/eval` dashboard, streaming chat mid-tool-call
- Pin a "demo path" — exact query, expected behaviour, what to look at
- Add "How to read this repo" section linking `plans/`, `docs/decisions/`, and eval results
- Link `plans/decisions/retrieval-finding.md` once written

### 9i — Repo hygiene

- Verify `.DS_Store`, `.pycache/`, `.ruff_cache/`, `.pytest_cache/` are in `.gitignore`
- Remove any tracked `.DS_Store` files
- Audit `data-set/` — document why it's committed or gitignore it
- Add pre-commit hook to block `.DS_Store` and `__pycache__` commits

### 9j — AI-pairing disclosure

- Short "How this was built" section in README near the top: pair-programmed with Claude Code; architecture, phase scoping, eval design, and ship decisions owned by Himanshu
- Point to `plans/` and `docs/improvement-log.md` as decision-making artefacts

### 9k — Measure and reconsider `_infer_category`

- `repository.py:75-102` is a hand-rolled keyword classifier used by `hybrid+rerank+filter`; it is the worst performer in the Phase 6 table
- Measure: turn the filter off across the whole eval set; record P@3/NDCG delta
- If gain negligible: delete `_infer_category` and the `enable_metadata_filter` flag
- If gain real: replace with a Claude one-shot classifier call (cached per session) and re-measure
- Update `thresholds.json` `best_mode` based on the corrected ranking

Success signal:

- `plans/decisions/retrieval-finding.md` written and linked from README
- Eval metric is chunk-aware; NDCG@5, MRR, hit@k reported in benchmark
- KB grown to ≥ 15 docs; eval set grown to ≥ 60 queries
- Adversarial eval set added; refusal metrics gated in CI
- Pareto chart embedded in `docs/benchmark.md`
- README reflects current state with dashboard screenshot and AI-pairing disclosure
- `_infer_category` measured, then replaced or removed
