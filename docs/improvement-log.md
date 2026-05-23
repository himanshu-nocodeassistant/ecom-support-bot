# Improvement Log

Use this file after each release. Keep it short and honest.

---

## Phase 5 — Evaluation & Retrieval Showcase

### Release

- Version: `v0.5.0-evaluation`
- Date: 2026-05-23

### What changed

- Added: `backend/eval/queries.json` — 30 labeled queries across 7 categories (refund, shipping, product, order, multi-intent, off-topic, unanswerable); each row has `expected_source_title` and `acceptable_answer_keywords`
- Added: `backend/eval/run.py` — full eval runner with `--mode`, `--all-modes`, `--chunking-audit` flags; metrics: precision@3, recall@3, context relevance, latency p50/p95, estimated Voyage cost; writes structured JSON to `backend/eval/results/`
- Added: `chunk_strategy` param on `chunk_knowledge_documents` and `import_knowledge_to_postgres` — `"fixed"` (220-char accumulation) and `"semantic"` (one paragraph per chunk)
- Added: `plans/decisions/chunking.md` — decision doc for fixed vs semantic chunking; benchmark via `--chunking-audit`
- Added: `backend/sql/migrate_5c_metadata.sql` — adds `doc_type`, `date_updated` to `knowledge_documents`; GIN index on `metadata->>'category'`
- Added: `deduplicate_chunks()` in `data_loader.py` — drops near-duplicate chunks (cosine similarity ≥ 0.95) at index time; logs count to stderr; returned in import result dict
- Added: `enable_reranking` and `enable_metadata_filter` flags on `PostgresRepository` — Voyage `rerank-2-lite` as post-retrieval step; keyword-based `_infer_category()` for pre-filtering; both off by default, toggled per eval mode
- Added: `_extract_sources()` in `agent.py` — collects cited source titles from `search_knowledge_base` tool events; suppresses scores < 0.15; attaches `sources` array to every API response and SSE `done` event
- Added: source chips in `frontend/index.html` — purple chips rendered below each bot reply showing title and score
- Added: `backend/app/eval_dashboard.py` — server-side HTML generator reading `backend/eval/results/` JSON files
- Added: `GET /eval` route on FastAPI — color-coded summary table, p50/p95 latency bar chart, per-query drill-down for best mode
- Improved: `KnowledgeDocument` carries `doc_type` and `date_updated`; every chunk JSONB metadata includes `doc_type`, `source_section`, `chunk_strategy`
- Improved: eval modes expanded to `keyword`, `fulltext`, `hybrid`, `hybrid+rerank`, `hybrid+rerank+filter`

### Why it matters

- User impact: every bot response now shows which source documents were used; source chips give transparent grounding, not a black-box answer
- Engineering impact: retrieval quality is now a number, not a demo — `--all-modes` produces a side-by-side comparison table; the `/eval` dashboard makes regressions visible before they ship; deduplication keeps the index clean across re-imports

### What is still weak

- Session memory is still in-process — server restart loses all sessions
- Answer correctness metric requires a running Claude API call per query to be meaningful; current implementation is keyword-overlap only
- Eval covers retrieval quality, not end-to-end agent quality (tool selection, multi-step correctness)

---

## Phase 4 — Streaming Product Experience

### Release

- Version: `v0.4.0-streaming-ui`
- Date: 2026-05-23

### What changed

- Added: `POST /api/chat/stream` — SSE endpoint that emits `tool_start`, `tool_result`, `token`, `done`, `error` events as the agent loop runs
- Added: `POST /api/compare` — runs phase1, phase2, phase3 concurrently via `asyncio.gather` and returns all three responses in one call
- Added: `frontend/index.html` — single-file chat UI with Chat tab (streaming) and Compare tab (three-column phase comparison); no build tooling
- Added: `mode` param on `POST /api/chat` — switches between `phase1`, `phase2`, `phase3`, `phase4` without restarting the server
- Added: `_repo_for_mode` in `agent.py` — selects the right repository (in-memory vs Postgres+Voyage) based on mode
- Added: CORS middleware so the HTML file can talk to the API from any origin
- Added: 17 tests covering all phase modes, compare endpoint, and SSE event format

### Why it matters

- User impact: tool activity is now visible as it happens — users see lookup_order and request_refund cards appear before the reply arrives; perceived latency on multi-tool queries drops significantly
- Engineering impact: phase comparison is a single API call; any query can be run against all three retrieval and routing strategies simultaneously; the compare tab makes the cumulative improvement visible in one view

### What is still weak

- Session memory is still in-process — server restart loses all sessions
- Compare tab runs phase1/2/3 only; phase4 streaming doesn't participate by design (compare needs synchronous responses)
- No evaluation layer — improvements across phases are demonstrated by example, not measured

---

## Phase 3 — Agent Tool Loop

### Release

- Version: `v0.3.0-agent-tool-loop`
- Date: 2026-05-23

### What changed

- Added: Claude API agentic loop in `agent.py` — `handle_message` now sends full conversation history + four tool schemas to `claude-haiku-4-5-20251001` and runs until `stop_reason == "end_turn"`
- Added: Formal JSON tool schemas for `lookup_order`, `request_refund`, `search_knowledge_base`, `create_ticket`
- Added: `_execute_tool` dispatcher — routes Claude's tool calls to the existing tool functions
- Added: `ANTHROPIC_API_KEY` to `Settings`; `anthropic>=0.40` to requirements
- Improved: session memory now stores full Anthropic-format message history (including tool calls and results), giving Claude complete context across turns
- Improved: ticket subjects are model-generated and contextual rather than generic
- Improved: refund refused for undelivered orders now offers alternatives instead of failing silently
- Kept: deterministic routing fallback when no API key is set; all existing tests pass without an API key

### Why it matters

- User impact: multi-step flows (look up order, validate delivery, process refund) now complete in a single user message; "Can I get my money back?" resolves via natural conversation rather than escalating immediately
- Engineering impact: routing logic is no longer maintained in code — adding a new tool only requires a new JSON schema and an executor function; the model handles orchestration

### What is still weak

- No streaming — user waits for the full agentic loop before seeing any output
- No UI — raw JSON API only
- Tool errors are passed back to Claude as JSON strings with no structured retry logic
- Session memory is in-process; a server restart loses all sessions

---

## Phase 2 — Retrieval Upgrade

### Release

- Version: `v0.2.0-retrieval-upgrade`
- Date: 2026-05-21

### What changed

- Added: `PostgresRepository` with full-text search (`tsvector`) and hybrid search (30% FTS + 70% cosine similarity via Voyage AI `voyage-3-lite`)
- Added: `knowledge_chunks` table with `vector(512)` column and HNSW index (`vector_cosine_ops`)
- Added: `embed_texts` / `embed_query` helpers in `data_loader.py`; `--voyage-api-key` flag on `import-knowledge` CLI
- Added: `compare-retrieval` CLI command for side-by-side FTS vs hybrid comparison
- Improved: confidence threshold raised from `0.05` to `0.25` to account for continuous hybrid scores
- Improved: "Can I get my money back?" now returns refund policy (score 0.266) instead of escalating
- Improved: "How long until my package shows up?" now returns shipping policy (score 0.273) instead of escalating

### Why it matters

- User impact: queries with no keyword overlap with the knowledge base now resolve correctly — the three most common semantic miss cases all fixed
- Engineering impact: retrieval quality is now measurable (continuous float scores vs integer keyword counts); the fallback chain (hybrid → FTS → in-memory) keeps tests fast without external deps

### What is still weak

- Routing is still deterministic if/elif — no LLM in the loop
- Replies are template strings, not model-generated
- The 0.25 threshold is a heuristic; edge cases near the boundary require manual review

---

## Phase 1 — Thin Vertical Slice

### Release

- Version: `v0.1.0-core`
- Date: 2026-05-18

### What changed

- Added: Supabase schema, repository layer, Olist loader CLI, Postgres import path
- Improved: order lookup now works with real Olist IDs and database-backed records
- Improved: knowledge retrieval can now read persisted documents and chunks from Supabase
- Removed: hard dependency on in-memory orders for order status checks

### Why it matters

- User impact: order questions can now resolve against real imported data instead of only demo IDs
- Engineering impact: storage and agent logic are decoupled, making retrieval and tool upgrades much easier

### What is still weak

- Knowledge retrieval is persisted but still keyword/full-text based
- Refund/ticket flows are deterministic and not model-driven

---

## Template

### Release

- Version:
- Date:

### What changed

- Added:
- Improved:
- Removed:

### Why it matters

- User impact:
- Engineering impact:

### What is still weak

- Weakness 1:
- Weakness 2:

### Next release focus

- Next target:
- Success signal:
