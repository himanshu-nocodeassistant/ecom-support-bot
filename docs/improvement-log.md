# Improvement Log

Use this file after each release. Keep it short and honest.

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
