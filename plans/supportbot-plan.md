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

- [ ] Run `python -m backend.eval.run --all-modes` against live Supabase+Voyage stack
- [ ] Commit raw JSON results to `backend/eval/results/`
- [ ] Generate `docs/benchmark.md` — markdown comparison table (mode × metric) committed to repo
- [ ] This snapshot becomes the regression baseline for all future changes

---

### 6b: LLM-Judge Answer Correctness

- [ ] Replace keyword-overlap correctness with a Claude API judge call per query
- [ ] Prompt: given query + retrieved context + answer, score factual accuracy 0–1
- [ ] Add `answer_correctness_llm` column alongside existing keyword score for comparison
- [ ] Record cost per query for the judge call

---

### 6c: Agent Quality Fixtures

- [ ] 10–15 scripted multi-turn conversation fixtures with expected tool sequences
- [ ] Metrics: tool selection accuracy, unnecessary tool calls, correct refusals
- [ ] Stored as `backend/eval/agent_fixtures.json`
- [ ] Runner: `python -m backend.eval.run --agent-eval`

---

### 6d: Regression Gate

- [ ] GitHub Actions workflow: runs eval on every PR against main
- [ ] Fails CI if any metric drops > 10% from committed baseline
- [ ] Posts metric delta as PR comment
- [ ] Config: thresholds stored in `backend/eval/thresholds.json`

---

### Acceptance criteria

- [ ] `docs/benchmark.md` exists with real numbers from the live stack
- [ ] Answer correctness uses LLM-judge, not keyword overlap
- [ ] Agent fixture eval covers ≥ 10 multi-turn scenarios
- [ ] CI fails on metric regression
