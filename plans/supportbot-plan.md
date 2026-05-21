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

- [ ] Product questions can be answered from an in-repo knowledge base
- [ ] Order status questions work for sample order IDs
- [ ] Session context remembers a prior order ID for follow-up questions
- [ ] Low-confidence queries create a support ticket instead of bluffing
- [ ] A new developer can run the project locally with simple setup

---

## Phase 2: Retrieval Upgrade

**User stories**: 1, 6, 11, 12, 16, 17

### What to build

Replace keyword retrieval with proper chunking, metadata preservation, embeddings, and hybrid search. Expose confidence signals so the agent can decline weak answers more reliably.

### Acceptance criteria

- [ ] Knowledge documents are chunked with metadata
- [ ] Vector and full-text search both contribute to retrieval
- [ ] Confidence scoring is visible and affects fallback behavior
- [ ] Query quality improves on known exact-match and semantic cases

---

## Phase 3: Agent Tool Loop

**User stories**: 2, 3, 6, 7, 18, 20

### What to build

Move from deterministic routing to a model-driven tool loop that can chain order lookup, refund logic, and escalation in one conversational flow.

### Acceptance criteria

- [ ] Tool definitions are formalized and validated
- [ ] Multi-step queries can trigger multiple tool calls
- [ ] Refund flows validate eligibility and return references
- [ ] Conversation memory supports follow-up queries without repeated identification

---

## Phase 4: Streaming Product Experience

**User stories**: 5, 9, 12, 13

### What to build

Add a real frontend, stream responses progressively, and show the user what the system is doing while it searches or uses tools.

### Acceptance criteria

- [ ] The chat UI streams assistant output
- [ ] Tool activity is visible in the interface
- [ ] Interaction logs and metrics are attached to each run
- [ ] The project feels like a coherent product, not just an API

---

## Phase 5: Showcase And Evaluation

**User stories**: 11, 13, 15

### What to build

Add the educational retrieval comparison showcase, plus a repeatable evaluation layer that makes each upgrade measurable.

### Acceptance criteria

- [ ] Baseline and upgraded retrieval modes can be compared side by side
- [ ] Evaluation queries demonstrate measurable improvement after each release
- [ ] The repo clearly explains what changed, why it matters, and what comes next
