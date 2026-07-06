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

## `v0.9.0-rag-honesty-and-polish` ✅ Complete

Resolved the keyword-beats-hybrid finding from the Phase 6 baseline and brought eval coverage, README, and repo hygiene up to current state.

- **9a — Retrieval finding:** `_precision_at_k` had a structural bias against chunked modes — a correct chunked hit only scored 1/3 if other slots pulled different docs. Rebuilt metrics on `document_id`; re-ran all modes; root-cause documented in `plans/decisions/retrieval-finding.md`; `thresholds.json` `best_mode` updated
- **9b — KB expansion:** grew to 15 docs (returns, warranty, payment, account management, order modification, address change, gift wrapping, subscription billing, B2B/wholesale, accessibility, 3 product guides); `queries.json` extended to 62 labeled queries
- **9c — Eval metric fixes:** doc-ID P@3/R@3 replacing title-string metric; `hit_rate_at_k` for k ∈ {1,3,5,10}; NDCG@5; MRR; title-based columns kept for historical comparison
- **9d — Synthetic query generation:** `generate_queries.py` produces 5 paraphrased + 2 adversarial per doc; embedding-based dedup; `--query-set gold|synthetic|both`
- **9e — Adversarial eval:** 40 queries (10 injection, 10 ambiguous, 10 multi-intent, 10 OOS); `injection_refusal_rate`, `clarification_rate`, `multi_tool_rate`, `oos_refusal_rate`; CI gate ≥ 0.80
- **9f — Pareto visualisation:** cost vs NDCG@5 and latency vs NDCG@5 SVGs embedded in `docs/benchmark.md`, regenerated by `--benchmark`
- **9g — Benchmark trend history:** `docs/benchmark-history.jsonl` + sparklines; fingerprinted by `{n_docs, metric_version}` so incomparable runs don't connect
- **9h/9j — README + AI-pairing disclosure:** capabilities table refreshed; "How this was built" section added; demo path pinned; eval commands documented
- **9i — Repo hygiene:** `.DS_Store`, `__pycache__`, `.ruff_cache` in `.gitignore`; pre-commit hooks block future commits of those files; `ruff` + `ruff-format` enforced
- **9k — `_infer_category` deleted:** measured 4.3% hard-miss rate on multi-intent queries; `enable_metadata_filter` flag removed; `hybrid+rerank` replaces `hybrid+rerank+filter` as best mode

What remains weak:

- Run-to-run variance in hybrid eval results (root-caused in v0.9.1 as silent Voyage rate-limit degradation)
- Eval layer had several measurement bugs not caught until a pre-publication audit (fixed in v0.9.1)

---

## `v0.9.1-eval-audit` ✅ Complete

Pre-publication audit of the eval layer. Running the eval suite consistently and publishing the numbers required catching and fixing five broken instruments, correcting two contaminated decisions, and stating one scope boundary.

**Five broken instruments fixed:**

- **Fail-open regression gate** — `check_regression.py` silently `continue`d when a gated metric was missing from the baseline; retrieval quality could drop to zero and CI would pass. Fixed: `--strict` flag exits 1 on any skipped metric; guard test `test_baseline_covers_every_gated_metric` prevents recurrence
- **ID namespace mismatch** — in-memory store returned ids like `kb-refund`; `queries.json` expected `refund-policy`; keyword mode scored 0.000 on all doc-id metrics despite correct retrieval. Fixed: ids aligned; guard test `test_expected_doc_ids_exist_in_inmemory_kb` catches future drift
- **Refusal oracle** — `_REFUSAL_KEYWORDS` included bare words like `"not"` and `"only"`; any reply containing "not" (e.g. "I could not find your order, but here is your refund confirmation") scored as a correct refusal. Fixed: scored from structured signal — absence of `request_refund` call — not string matching
- **Silent hybrid degradation** — `embed_query()` had no retry unlike its siblings; a Voyage 429 caused `_hybrid_search` to fall back to fulltext-only silently, producing run-to-run variance. Fixed: `embed_query` retries via `embed_queries`; fallback logs a warning and tags results `degraded`; `--strict` aborts on any degraded query in CI
- **Tautological faithfulness** — `evaluate_faithfulness` was fed `answer=retrieved[0].content`, judging a chunk against a context containing itself. No honest rename exists. Removed from `run.py` and tests

**Two contaminated decisions corrected:**

- **Chunking +29% semantic-wins claim** — measured with `_precision_at_k`, the same biased metric. Re-run on doc-id `hit_rate@3`: fixed 0.346 vs semantic 0.327 — no strong evidence either strategy wins. Claim retired; decision doc updated
- **Reranking delta** — prior numbers (hybrid NDCG@5=0.253 → rerank 0.288) were measured on a partially degraded hybrid baseline. Re-run on confirmed-undegraded runs: hybrid 0.934 → rerank 0.960. Decision doc updated

**Additional fixes:**

- H@5/H@10/NDCG@5 backed by real eval-only deeper retrieval (was re-sliced top-3, making H@5 = H@10 = H@3 by construction)
- `CtxRel` reported as `—` for non-embedding modes instead of `0.000`
- KwCorr documented as same-mode-only; `avg_kw_context_chars` added so text-volume gaps are legible
- LLM-judge renamed from "answer correctness" to "context relevance" — it judges the top retrieved chunk, not a generated answer
- Order-lookup stochastic-slice gap closed: bare numeric IDs, multi-turn unknown orders, order IDs in natural prose
- Eval boundary stated in README: what the suite covers and what it doesn't
- Benchmark history fingerprinted so incomparable runs don't produce fake trend lines
- `LICENSE` (MIT) and `.env.example` added
- README §2–§5: new opener, "Why I built this", Measured decisions table with verified numbers, "I" voice, Known gaps rewrite, footer
