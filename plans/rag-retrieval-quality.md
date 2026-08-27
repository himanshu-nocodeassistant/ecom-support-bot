# Plan: Lean RAG Retrieval Quality

> Source PRD: Conversation-approved RAG improvement scope, 2026-08-27

## Architectural decisions

Durable decisions that apply across all phases:

- **Routes**: Keep the existing chat and streaming API contracts. Retrieval improvements remain internal.
- **Storage**: Keep PostgreSQL, pgvector, and PostgreSQL full-text search. Do not add another vector database.
- **Retrieval depth**: Separate candidate depth from final context depth. Fetch 15–20 candidates and return at most 3 final results.
- **Retrieval modes**: Keep the current weighted hybrid mode as the baseline. Add RRF as an experiment and adopt it only if measured results improve.
- **Key models**: Use a common retrieval-candidate model that can carry document metadata, stage ranks, stage scores, and degraded-state information.
- **Reranking**: Keep Voyage as the reranking provider. Reranking receives the larger fused candidate set, not the final top 3.
- **Observability**: Langfuse is optional and fail-open. A tracing outage must not affect customer requests.
- **Privacy**: Do not send raw customer email, order IDs, or other direct identifiers to Langfuse.
- **Evaluation**: Keep versioned local evaluation data and CI gates as the source of truth. Langfuse provides trace analysis and experiment views, not the only copy of results.
- **Frameworks**: Do not add LangChain or LangGraph in this plan.

---

## Phase 1: Deep-Candidate Reranking

**User stories**: As a customer, I want relevant evidence to remain eligible for reranking even when it is outside the first three search results, so that I receive a better-supported answer. As a developer, I want candidate depth and final context depth to be separate settings, so that I can tune quality without increasing prompt size.

### What to build

Extend the existing hybrid retrieval path to fetch a larger candidate set, send that set to Voyage reranking, and return only the best three results to the agent. Preserve the existing API response shape and degraded fallback behavior. Add benchmark output that records candidate depth, final depth, retrieval quality, latency, and estimated cost.

### Acceptance criteria

- [ ] The production retrieval path can fetch 15–20 candidates and still returns at most 3 final results.
- [ ] Voyage receives more than 3 candidates when reranking is enabled and enough candidates exist.
- [ ] Candidate depth and final depth can be changed independently.
- [ ] Existing fallback behavior remains visible and does not silently report a degraded run as a clean run.
- [ ] Automated tests verify public retrieval behavior, including shallow corpora with fewer candidates than requested.
- [ ] A benchmark compares the current top-3 reranking path with deep-candidate reranking using quality, latency, and cost metrics.

---

## Phase 2: RRF Retrieval Experiment

**User stories**: As a developer, I want lexical and semantic retrieval to produce independent ranked lists, so that one score scale cannot dominate the other. As a product owner, I want RRF adopted only when it produces a measured improvement, so that the system does not gain complexity without value.

### What to build

Add an experimental retrieval mode that runs PostgreSQL full-text search and pgvector search independently, merges their candidate ranks with reciprocal rank fusion, reranks the fused candidates, and returns the best three. Compare this mode with the current weighted hybrid baseline on the same fixed dataset. Keep the measured winner as the recommended mode.

### Acceptance criteria

- [ ] Sparse and dense searches each return an independent ranked candidate list.
- [ ] Fusion handles candidates found by one or both search paths without duplicate final entries.
- [ ] Fused candidates retain enough stage metadata to explain their sparse, dense, and fused ranks.
- [ ] RRF and weighted hybrid use the same candidate depth, reranker, final depth, dataset, and evaluation method.
- [ ] The comparison reports retrieval quality, latency, cost, and degraded-query count.
- [ ] RRF becomes the recommended mode only if it clears the existing regression gates and improves the agreed primary quality metric without an unacceptable latency or cost increase.
- [ ] Automated tests verify fusion ordering, duplicate handling, missing-list behavior, and deterministic tie handling through observable results.

---

## Phase 3: Minimal Langfuse Tracing

**User stories**: As a developer, I want to inspect one customer request from query through retrieval, reranking, generation, and tools, so that I can diagnose quality and latency failures. As an operator, I want tracing failures isolated from customer traffic, so that observability cannot make support unavailable.

### What to build

Add optional Langfuse tracing around the full chat request. Record separate observations for sparse retrieval, dense retrieval, fusion, reranking, generation, and tool calls. Attach safe configuration metadata, ranks, latency, token use, cost, and degraded-state information. Exclude direct customer identifiers and allow tracing to be disabled through configuration.

### Acceptance criteria

- [ ] One chat request produces a trace with retrieval, reranking, generation, and tool observations when those stages run.
- [ ] Retrieval observations include candidate depth, final depth, candidate document IDs, stage ranks, latency, and degraded state.
- [ ] Generation observations include model, token use, latency, and estimated cost when the provider returns them.
- [ ] Session correlation uses a safe identifier that does not expose customer email or order IDs.
- [ ] Missing credentials, network failures, and Langfuse errors do not change the API response or prevent a request from completing.
- [ ] Tracing can be disabled without changing application code.
- [ ] Automated tests verify emitted trace metadata through a fake tracing boundary and verify fail-open behavior.

---

## Phase 4: End-to-End RAG Evaluation

**User stories**: As a product owner, I want to know whether final answers are supported by retrieved evidence, so that retrieval gains translate into customer value. As a developer, I want unsupported questions and incorrect citations covered by regression tests, so that the bot does not improve ranking metrics while becoming less trustworthy.

### What to build

Extend the existing evaluation suite from retrieval-only checks to complete generated answers. Add fixed cases with reference claims, expected supporting documents, and unsupported questions. Score answer faithfulness, citation accuracy, and correct refusal. Keep deterministic checks in CI and make model-judged evaluation an explicit optional run. Send experiment scores to Langfuse when it is configured.

### Acceptance criteria

- [ ] Evaluation cases can define reference claims, one or more expected supporting documents, and whether the question is answerable.
- [ ] The evaluation runs the same retrieval and generation path used by the chat API.
- [ ] Faithfulness measures whether answer claims are supported by retrieved content.
- [ ] Citation accuracy measures whether cited documents support the related answer claims.
- [ ] Unsupported-question evaluation distinguishes a correct refusal or escalation from an unsupported answer.
- [ ] Deterministic metrics run without Langfuse and without an LLM judge.
- [ ] Model-judged metrics record the judge model, prompt version, dataset version, and run time.
- [ ] CI fails when agreed end-to-end thresholds regress.
- [ ] Langfuse receives experiment scores when configured, while local result files remain complete and authoritative.

---

## Out of scope

- LangChain or LangGraph adoption
- Prompt management
- A new vector database or search service
- User-interface changes
- Authentication, rate limiting, and unrelated production hardening
- Automatic query decomposition or dynamic retrieval depth
- Context compression and parent-document retrieval
- Replacing Voyage embeddings or reranking
