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

## `v0.4.0-streaming-ui` — Next

Improve the product feel:

- Frontend chat interface
- Streaming output (SSE or WebSocket)
- Tool activity timeline visible in the UI
- Better UX states (loading, error, escalation)

Success signals:

- First token appears in under 500ms for a single-tool query
- Tool calls are visible as they happen, not revealed after the fact
- The project feels like a coherent product, not an API

---

## `v0.5.0-showcase-and-evaluation`

Add the educational layer and make each upgrade measurable:

- Side-by-side retrieval comparison UI (keyword vs hybrid)
- Repeatable evaluation query set with ground-truth labels
- Score deltas between phases measured and displayed
- The repo explains what changed, why it matters, and what comes next

Success signals:

- A new reader can understand the full system evolution from the README alone
- Evaluation scores improve monotonically Phase 1 → Phase 2 → Phase 3
