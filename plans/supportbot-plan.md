# Plan: SupportBot

## Architectural decisions

Durable decisions that should survive across releases:

- **Product shape**: customer support assistant for a realistic e-commerce flow
- **Core surfaces**: chat experience, order lookup, refund flow, escalation flow, educational showcase
- **Backend boundary**: Python service owns agent orchestration, retrieval, tools, and session memory
- **Frontend boundary**: presentation layer consumes backend chat and observability events
- **Data shape**: structured order data plus unstructured support/product knowledge
- **Retrieval target state**: start with keyword retrieval, evolve to semantic chunking, then hybrid retrieval, then re-ranking
- **Tooling target state**: deterministic local tools first, then model-driven tool orchestration
- **Release strategy**: each phase should be demoable, pushable to GitHub, and accompanied by an improvement note

---

## Phase 1: Thin Vertical Slice

**User stories**: 1, 2, 4, 8, 10, 14

### What to build

A minimal but working support agent that can answer a few product questions from a small knowledge base, look up sample orders, remember session context, and escalate low-confidence requests into tickets.

### Acceptance criteria

- [x] Product questions can be answered from an in-repo knowledge base
- [x] Order status questions work for sample order IDs
- [x] Session context remembers a prior order ID for follow-up questions
- [x] Low-confidence queries create a support ticket instead of bluffing
- [x] A new developer can run the project locally with simple setup

---

## Phase 2: Retrieval Upgrade

**User stories**: 1, 6, 11, 12, 16, 17

### What to build

Replace keyword retrieval with proper chunking, metadata preservation, embeddings, and hybrid search. Expose confidence signals so the agent can decline weak answers more reliably.

### Acceptance criteria

- [x] Knowledge documents are chunked with metadata
- [x] Vector and full-text search both contribute to retrieval
- [x] Confidence scoring is visible and affects fallback behavior
- [x] Query quality improves on known exact-match and semantic cases

---

## Phase 3: Agent Tool Loop

**User stories**: 2, 3, 6, 7, 18, 20

### What to build

Move from deterministic routing to a model-driven tool loop that can chain order lookup, refund logic, and escalation in one conversational flow.

### Acceptance criteria

- [x] Tool definitions are formalized and validated
- [x] Multi-step queries can trigger multiple tool calls
- [x] Refund flows validate eligibility and return references
- [x] Conversation memory supports follow-up queries without repeated identification

---

## Phase 4: Streaming Product Experience

**User stories**: 5, 9, 12, 13

### What to build

Add a real frontend, stream responses progressively, and show the user what the system is doing while it searches or uses tools.

### Acceptance criteria

- [x] The chat UI streams assistant output
- [x] Tool activity is visible in the interface
- [x] Interaction logs and metrics are attached to each run
- [x] The project feels like a coherent product, not just an API

---

## Phase 5: Evaluation & Retrieval Showcase

**User stories**: 11, 13, 15

### What to build

A repeatable evaluation harness with a small dashboard that makes retrieval quality measurable and visible. Covers the full retrieval research agenda — chunking strategy, metadata, deduplication, advanced retrieval techniques, and LLM output quality — so every upgrade can be justified with numbers, not intuition.

---

### 5a: Ground Truth Dataset

Before any metric can be computed, a labeled eval set must exist.

- [x] 25–30 labeled queries covering refund, shipping, product, order, and off-topic categories
- [x] Each row: `query`, `expected_source_title`, `acceptable_answer_keywords`, `category`
- [x] Stored as a versioned JSON or CSV file in `backend/eval/queries.json`
- [x] Includes edge cases: paraphrased queries, multi-intent queries, unanswerable queries

---

### 5b: Chunking Strategy Audit

- [x] Document the current fixed-size chunking parameters (chunk size, overlap)
- [x] Implement semantic chunking (split on sentence/paragraph boundaries) as an alternative
- [x] Run eval dataset against both strategies and record precision@3 for each
- [x] Record decision: which strategy wins and why, committed to `plans/decisions/chunking.md`

---

### 5c: Metadata & Table Extraction

- [x] Audit all knowledge documents for structured content (tables, lists, key-value pairs)
- [x] Extract and store metadata fields per chunk: `doc_type`, `category`, `source_section`, `date_updated`
- [x] Add table-aware extraction for any documents containing tabular data (e.g. shipping zones, pricing tiers)
- [x] Metadata stored as JSONB column on `knowledge_chunks` table

---

### 5d: Deduplication

- [x] Detect near-duplicate chunks using cosine similarity threshold (≥ 0.95 = duplicate)
- [x] Deduplication pass runs at index time in `data_loader.py`, not at query time
- [x] Log how many chunks were dropped per ingest run

---

### 5e: Advanced Retrieval Techniques

Each technique is behind a feature flag so they can be toggled for A/B comparison.

- [x] **Metadata filtering** — pre-filter chunks by `category` before vector search when query intent is classifiable
- [x] **Reranking** — add Voyage AI rerank API call as a post-retrieval step; compare top-3 results before and after
- [ ] **Query expansion** — deferred (high latency cost, marginal gain at current KB size)
- [ ] **Multi-query retrieval** — deferred
- [ ] **Parent-child retrieval** — deferred
- [ ] **Context compression** — deferred

---

### 5f: Citation & Source Grounding

- [x] Agent responses cite the source document title and category inline
- [x] Each API response includes a `sources` array: `[{title, category, score}]`
- [x] UI displays source chips below assistant responses
- [x] Low-confidence sources (score < 0.15) are suppressed from citations

---

### 5g: Evaluation Metrics

Run against the labeled query set from 5a. All metrics computed per retrieval mode.

- [x] **Precision@3** — fraction of top-3 retrieved chunks that are relevant
- [x] **Recall@3** — fraction of relevant chunks found within top-3
- [x] **Context relevance score** — average cosine similarity between query embedding and retrieved chunk embeddings
- [x] **Answer correctness** — keyword overlap between generated answer and `acceptable_answer_keywords`
- [x] **Latency** — p50/p95 end-to-end response time per retrieval mode, measured in the eval runner
- [x] **Cost per query** — Voyage API token usage estimated per mode
- [x] All metrics emitted as structured JSON to `backend/eval/results/`

---

### 5h: A/B Retrieval Comparison

- [x] Eval runner accepts a `--mode` flag: `keyword`, `fulltext`, `hybrid`, `hybrid+rerank`, `hybrid+rerank+filter`
- [x] Single command runs all modes against the same query set and writes a comparison table
- [x] Results include: mode, precision@3, recall@3, context relevance, avg latency, avg cost

---

### 5i: Evaluation Dashboard

A minimal read-only web view rendered as a `/eval` route on the existing FastAPI backend.

- [x] Summary table: all retrieval modes × all metrics, color-coded (green = best, red = worst)
- [x] Per-query drill-down: for the best mode, show retrieved titles, P@3, R@3 per query
- [x] Latency distribution chart (p50/p95 bar chart per mode)
- [x] Cost breakdown per mode

---

### Considered and deferred

The following techniques were evaluated and deferred — not because they're unimportant, but because their value is limited at the current knowledge base size and scope:

- **Parent-child retrieval** — deferred until chunk granularity becomes a real context-loss problem
- **Context compression** — deferred until prompt context limits are actually hit
- **Query expansion / multi-query** — high latency cost for marginal gain on a small KB; revisit if KB grows past 500 chunks
- **Full BM25** — PostgreSQL `ts_rank` is used instead of true BM25; gap is negligible at this scale

---

### Acceptance criteria

- [x] Labeled eval dataset exists with ≥ 25 queries
- [x] At least 3 retrieval modes are benchmarked head-to-head
- [x] All 6 metrics (precision, recall, context relevance, correctness, latency, cost) are reported per mode
- [x] Evaluation dashboard renders the comparison in a browser (`GET /eval`)
- [x] Citation grounding is live in the chat UI
- [x] One-command eval run: `python -m backend.eval.run --all-modes`

---

## Phase 6: Data-Backed Benchmark

**Goal**: make every improvement verifiable with committed numbers — not prose claims. Readers should be able to see the exact metric delta each change produced.

---

### 6a: Baseline Benchmark Run

- [x] Run `python -m backend.eval.run --all-modes` against live Supabase+Voyage stack
- [x] Commit raw JSON results to `backend/eval/results/`
- [x] Generate `docs/benchmark.md` — markdown comparison table (mode × metric) committed to repo
- [x] This snapshot becomes the regression baseline for all future changes

---

### 6b: LLM-Judge Answer Correctness

- [x] Replace keyword-overlap correctness with a Claude API judge call per query
- [x] Prompt: given query + retrieved context + answer, score factual accuracy 0–1
- [x] Add `answer_correctness_llm` column alongside existing keyword score for comparison
- [x] Record cost per query for the judge call

---

### 6c: Agent Quality Fixtures

- [x] 10–15 scripted multi-turn conversation fixtures with expected tool sequences
- [x] Metrics: tool selection accuracy, unnecessary tool calls, correct refusals
- [x] Stored as `backend/eval/agent_fixtures.json`
- [x] Runner: `python -m backend.eval.run --agent-eval`

---

### 6d: Regression Gate

- [x] GitHub Actions workflow: runs eval on every PR against main
- [x] Fails CI if any metric drops > 10% from committed baseline
- [x] Posts metric delta as PR comment
- [x] Config: thresholds stored in `backend/eval/thresholds.json`

---

### Acceptance criteria

- [x] `docs/benchmark.md` exists with real numbers from the live stack
- [x] Answer correctness uses LLM-judge, not keyword overlap
- [x] Agent fixture eval covers ≥ 10 multi-turn scenarios
- [x] CI fails on metric regression

---

## Phase 7: Persistent Customer Memory (data layer — complete)

Data contracts and in-memory implementations are tested and green.
Nothing is wired into the live agent yet.

**What exists:**
- `InMemoryCustomerStore` — identity upsert, session linking, memory facts (confidence-gated, one per type), order linkage (sliding window of 5)
- `InMemoryConversationStore` — turn persistence, max-window load
- `build_customer_context()` / `build_system_prompt()` — context injection helpers
- SQL migration `migrate_7_customer_memory.sql` — five new tables
- 31 passing tests

**What does not exist yet:** nothing is wired into `agent.py`; `SESSION_MEMORY` dict still in use; fact extraction never runs; `customer_email` not accepted by the API.


---

## Phase 8: Memory Wiring & Production Hardening

Wire Phase 7's data layer into the live agent: agent integration, Postgres backends, session-end fact extraction, memory surfaces in the UI and eval harness.

---

### 8a: Replace SESSION_MEMORY with ConversationStore

Current `SESSION_MEMORY` is a plain dict — lost on restart, no cross-process sharing.

- [ ] Inject `ConversationStore` into `handle_message` and `handle_message_stream` (default: `InMemoryConversationStore`)
- [ ] Replace all `SESSION_MEMORY.setdefault(...)` / `SESSION_MEMORY[session_id] = messages` with `store.save_turn()` / `store.load_turns()`
- [ ] Remove `SESSION_MEMORY` module-level dict entirely once tests pass
- [ ] All existing `test_support_bot.py` and `test_phase_modes.py` tests continue to pass (behaviour unchanged, only storage changes)

---

### 8b: Wire CustomerStore into handle_message

- [ ] `handle_message` and `handle_message_stream` accept optional `customer_email: str | None = None`
- [ ] When email is present: upsert customer, link session, load facts + prior orders
- [ ] Call `build_customer_context(facts, prior_order_ids)` and `build_system_prompt(customer_context)` to produce the injected system prompt
- [ ] After any `lookup_order` tool call succeeds for a known customer, call `store.link_order(customer_id, order_id)`
- [ ] Anonymous sessions (no email) are unchanged — no context injected, no orders linked
- [ ] New test class `CustomerAwareAgentTests` in `test_customer_memory.py`: assert that a known customer's prior order ID appears in the system prompt passed to Claude

---

### 8c: Session-end fact extraction

Extract facts once per session, not per turn. Runs as a FastAPI `BackgroundTask` so it does not block the chat response.

- [ ] `extract_session_facts(session_id, customer_id, conversation_store, customer_store)` — reads last N turns, calls Claude with extraction prompt, parses JSON response
- [ ] Extraction prompt: given the conversation, output `[{fact_type, fact_text, confidence}]` — zero items is valid
- [ ] Only `fact_type` values in `('order_preference', 'issue_history', 'product_interest', 'communication_style')` are accepted; anything else is discarded
- [ ] Each fact with `confidence >= 0.7` written via `customer_store.save_memory_fact()`
- [ ] Called from `POST /chat` as a background task after the response is sent
- [ ] Unit test: mock Claude response with known JSON → assert correct facts written, low-confidence facts discarded, unknown fact_types discarded

---

### 8d: PostgresConversationStore

- [ ] `PostgresConversationStore(database_url)` implementing `ConversationStore` protocol
- [ ] `save_turn`: `INSERT INTO conversation_turns (session_id, role, content) VALUES (%s, %s, %s)`
- [ ] `load_turns`: `SELECT role, content FROM conversation_turns WHERE session_id = %s ORDER BY created_at DESC LIMIT %s` — results reversed before returning (oldest first)
- [ ] Falls back to `InMemoryConversationStore` if `psycopg` import fails or DB unreachable
- [ ] Selected when `settings.data_backend == "postgres"` and `settings.database_url` is set

---

### 8e: PostgresCustomerStore

- [ ] `PostgresCustomerStore(database_url)` implementing `CustomerStore` protocol
- [ ] `upsert_customer`: `INSERT INTO customers (email, name) VALUES (%s, %s) ON CONFLICT (email) DO UPDATE SET name = EXCLUDED.name RETURNING *`
- [ ] `save_memory_fact`: `INSERT INTO customer_memory (...) ON CONFLICT (customer_id, fact_type) DO UPDATE SET fact_text = EXCLUDED.fact_text, confidence = EXCLUDED.confidence, updated_at = now() WHERE EXCLUDED.confidence >= customer_memory.confidence` — single atomic upsert, no race condition
- [ ] `get_customer_orders`: `SELECT order_id FROM customer_orders WHERE customer_id = %s ORDER BY linked_at DESC LIMIT 5`
- [ ] Falls back to `InMemoryCustomerStore` if DB unreachable

---

### 8f: API + Frontend

- [ ] `POST /chat` request body gains optional `customer_email: str | None` field
- [ ] `POST /chat/stream` same
- [ ] Frontend: optional email input above the chat textarea ("Sign in to save your history")
- [ ] If a recognized customer connects, display their name in the chat header ("Welcome back, Alice")
- [ ] No auth — email is taken at face value (trust model matches the demo scope)

---

### 8g: Memory decay

Facts written once live forever. A shipping preference from 6 months ago may be wrong today.

- [ ] Add `expires_at timestamptz null` column to `customer_memory` (migration)
- [ ] Default TTL: 90 days from `updated_at`
- [ ] `load_memory_facts` filters out expired rows (`WHERE expires_at IS NULL OR expires_at > now()`)
- [ ] `InMemoryCustomerStore` supports TTL check using `datetime.now()` at load time

---

### 8h: Concurrency safety

- [ ] Document the race condition: two concurrent sessions for the same customer can both attempt `save_memory_fact` for the same `fact_type`
- [ ] Postgres implementation is safe by design (atomic `ON CONFLICT DO UPDATE WHERE`)
- [ ] `InMemoryCustomerStore` noted as not thread-safe (acceptable for single-process dev use)
- [ ] Add a note to `customer_store.py` explaining why `PostgresCustomerStore` must be preferred in production

---

### 8i: Memory panel in eval dashboard

- [ ] New `/eval/memory` route or tab on the existing eval dashboard
- [ ] Table: `customer_id`, email, session count, fact count, last active
- [ ] Drill-down per customer: list of facts with `fact_type`, `fact_text`, `confidence`, `updated_at`

---

### 8j: Memory eval fixtures

- [ ] 5–8 multi-session conversation fixtures in `backend/eval/agent_fixtures.json`
- [ ] Each fixture: session 1 (customer identifies + has an issue), session 2 (customer returns, no re-identification)
- [ ] Expected: agent references prior context without prompting
- [ ] New metric: **memory recall rate** — fraction of fixtures where prior context surfaced
- [ ] Regression gate threshold: `memory_recall_rate >= 0.80`

---

### Acceptance criteria

- [ ] `SESSION_MEMORY` dict removed from `agent.py`
- [ ] `handle_message` accepts `customer_email` and injects memory context when present
- [ ] Fact extraction runs as a background task after each session reply
- [ ] `PostgresConversationStore` and `PostgresCustomerStore` implemented and selected automatically when DB is configured
- [ ] Frontend shows email input and welcome-back message
- [ ] Memory facts expire after 90 days
- [ ] Eval dashboard shows memory panel
- [ ] Memory recall rate metric in CI regression gate
