# Changelog

What each release added, why it mattered, and what was still weak when it went out.

Newest first. Each phase number matches its version tag, so "Phase 5" and `v0.5.0` are the same
release.

For the reasoning behind specific choices, see [`plans/decisions/`](../plans/decisions/). For work
that has not started yet, see [`plans/roadmap.md`](../plans/roadmap.md).

---

## Phase 9.2 — Eval contracts and safer multi-intent handling

`v0.9.2-eval-contracts` · 2026-07-15

### What changed

- Replaced the multi-intent test rule `called 2 or more different tools` with a written contract per
  scenario. Each one lists the tools it must call and the tools it must not call. It also says
  whether the agent has to ask a follow-up question or hand off to a human.
- Changed the prompt rules for mixed requests. Say a customer asks for one thing the bot can do and
  one thing it cannot. The bot now does the first and opens a ticket for the second. A refund with
  no stated reason gets a follow-up question instead of going through.
- Made the poisoned-knowledge safety test run offline. It no longer calls the live API just because
  a developer happens to have a key set.
- Added a stamp to every live eval result recording the dataset, the model, and the scoring version
  used. Strict mode now fails if a required result is missing or was made with different inputs.

### Why it matters

- For customers: a mixed request is less likely to come back as "all done" when half of it never
  happened. A refund with no reason given is less likely to go through.
- For the codebase: the gate now measures whether the agent resolved things safely, not how many
  tools it called. A skipped eval can no longer look green.

### What was still weak

- The new contract metric needs a fresh baseline from a live run. Old `multi_tool_rate` numbers are
  not comparable and should not be read as a trend.
- Prompt rules make the agent more consistent, but they are not enforcement. Actions that cannot be
  undone still need a server-side permission check, protection against running twice, and an audit
  record.

---

## Phase 9.1 — Eval audit

`v0.9.1-eval-audit` · 2026-07-06

Before publishing any numbers, I audited the eval layer itself. Five measurements turned out to be
broken, two past decisions rested on those broken measurements, and one gap in coverage had never
been written down.

### Five broken measurements

- **The regression gate could not fail.** When a gated metric was missing from the baseline,
  `check_regression.py` skipped it quietly. Retrieval quality could have dropped to zero and CI would
  still have passed. Fixed with a `--strict` flag that exits 1 on any skipped metric, plus a guard
  test so it cannot come back.
- **Keyword mode scored 0.000 on every document metric** because of a naming mismatch, not bad
  retrieval. The in-memory store returned ids like `kb-refund` while the query file expected
  `refund-policy`. Ids now match, with a guard test for future drift.
- **The refusal check counted the wrong thing.** It looked for words like "not" and "only" anywhere
  in the reply. So "I could not find your order, but here is your refund confirmation" scored as a
  correct refusal. It now reads a real signal: whether the refund tool actually ran.
- **Hybrid search was quietly falling back to keyword-only.** `embed_query()` had no retry. A
  rate-limit response from Voyage sent the query down the weaker path with no warning. That was the
  source of results that changed between runs. It now retries, logs a warning, tags the result as
  degraded, and `--strict` stops the run in CI.
- **The faithfulness score was circular.** It was handed the retrieved chunk as if it were the
  answer. So it checked a chunk against a context that already contained it. There is no honest way
  to rename that, so it was removed.

### Two decisions corrected

- **The "semantic chunking wins by 29%" claim was retired.** It came from the same biased metric the
  retrieval investigation had already thrown out. The bias happened to favour the winner. Re-run on
  document hit rate: fixed 0.346 vs semantic 0.327. No strong evidence either way, so no config
  change.
- **The reranking numbers were re-measured** on a run confirmed not to be degraded. Hybrid NDCG@5
  0.934 rising to 0.960 with rerank, and Hit@1 0.904 rising to 0.942. The old figures (0.253 to
  0.288) had been taken against a partly degraded baseline.

### Also in this release

- H@5, H@10 and NDCG@5 now come from a genuinely deeper retrieval pass. Before, every path returned
  at most three results, so H@5 and H@10 were the same number as H@3 by construction.
- Context relevance now reads `—` for modes with no embeddings, instead of a made-up `0.000`.
- The keyword-overlap score is documented as same-mode-only, with a character count next to it so
  the text-volume difference between modes is visible.
- The LLM judge was renamed from "answer correctness" to "context relevance". It scores the top
  retrieved chunk, not a written answer.
- Order lookup gained test coverage for bare numeric ids, repeat questions about an unknown order,
  and ids buried in ordinary sentences.
- The README now states what the eval suite covers and what it does not.
- Benchmark history is fingerprinted, so runs that cannot be compared no longer draw a fake trend
  line.
- Added `LICENSE` (MIT) and `.env.example`.

---

## Phase 9.0 — Honest RAG numbers and polish

`v0.9.0-rag-honesty-and-polish` · 2026-06-23

Chased down why keyword search appeared to beat hybrid search in the Phase 6 baseline. Also brought
eval coverage, the README, and repo hygiene up to date.

### What changed

- **The retrieval finding.** `_precision_at_k` was biased against chunked modes: a correct hit only
  scored 1/3 if the other two slots held different documents. Metrics were rebuilt on document id
  and every mode re-run. Written up in
  [`plans/decisions/retrieval-finding.md`](../plans/decisions/retrieval-finding.md).
- **A bigger knowledge base** — 15 documents covering returns, warranty, payment, account
  management, order changes, address changes, gift wrapping, subscription billing, wholesale,
  accessibility, and three product guides. The labeled query set grew to 62.
- **New metrics** — document-id P@3 and R@3, hit rate at 1, 3, 5 and 10, NDCG@5, and MRR. The old
  title-based columns stayed for historical comparison.
- **Generated queries** — `generate_queries.py` writes 5 rephrased and 2 tricky queries per
  document, then drops near-duplicates by embedding.
- **An adversarial set** — 40 queries split evenly across prompt injection, vague questions,
  multi-part requests, and out-of-scope asks, each gated in CI at 0.80.
- **Cost and latency plots** against NDCG@5, drawn into `docs/benchmark.md`.
- **Trend history** in `docs/benchmark-history.jsonl`, fingerprinted so runs with different document
  counts or metric versions do not connect.
- **Category pre-filtering was deleted.** It caused a 4.3% hard-miss rate on multi-part queries.

### Why it matters

- The headline comparison in the repo had been guarding the worst-performing mode. Fixing the metric
  changed which mode CI protects.
- A 62-query set across 15 documents gives the numbers enough ground to stand on to be worth
  publishing.

### What was still weak

- Hybrid results varied between runs. Root cause found in v0.9.1: silent rate-limit degradation.
- Several measurement bugs went unnoticed until the v0.9.1 audit.

---

## Phase 8 — Memory wiring

`v0.8.0-memory-wiring` · 2026-06-14

Connected the Phase 7 memory layer to the live agent. Phase 7 built the storage; this release made
it do something.

### What changed

- The `SESSION_MEMORY` dictionary was replaced by an injected `ConversationStore` in both agent
  handlers.
- `CustomerStore` was wired in. Given an email, the agent looks up or creates the customer, links
  the session, and puts their past orders and known facts into the system prompt.
- A successful order lookup now links that order id to the customer automatically.
- `fact_extractor.py` uses Claude Haiku to pull 1 to 3 facts from a finished conversation, keeping
  only those above a confidence threshold.
- `PostgresConversationStore` and `PostgresCustomerStore` were added, falling back to the in-memory
  versions when the database is unavailable.
- Memory facts expire after 90 days.
- `customer_email` was added to both chat endpoints, and the streaming `done` event now reports
  whether this is a returning customer.
- The frontend gained an email field, a welcome-back banner, and a linked indicator.
- A new `memory_recall_rate` metric with 7 multi-session fixtures, gated in CI at 0.75, plus an
  `/eval/memory` endpoint and dashboard panel.

### Why it matters

- For customers: someone who comes back does not have to explain themselves again. Past orders and
  preferences are already in front of the agent.
- For the codebase: conversation history survives a restart, and memory quality is now a number in
  CI rather than something to eyeball.

### What was still weak

- The email is taken at face value. There is no code or token to prove the person owns it, so
  anyone can type someone else's address and read their facts.
- Fact extraction runs on request rather than truly after a session ends.
- Postgres is only used when `DATABASE_URL` is set, and migrations have to be applied by hand.

---

## Phase 7 — Customer memory, storage layer

`v0.7.0-persistent-customer-memory` · 2026-05-29

Designed and tested customer memory in memory only. Nothing was connected to the live agent yet.

### What changed

- `conversation_store.py` — a `ConversationStore` interface and an in-memory version that appends
  turns and loads a limited window.
- `customer_store.py` — a `CustomerStore` interface and an in-memory version. It covers identity by
  email, session linking, facts, and a rolling list of the last 5 orders. Facts are stored one row
  per fact type, kept only above 0.7 confidence, and overwritten by a more confident version.
- `memory_context.py` — helpers that turn past orders and facts into a block of system prompt text.
- `migrate_7_customer_memory.sql` — five new tables, all safe to re-run, all indexed.
- Unit tests for the three new modules, plus backfilled tests for the Phase 6 eval code.
- `.github/workflows/test.yml` runs the test suite on every push and pull request.

### Why it matters

- For customers: nothing changed yet.
- For the codebase: memory has a settled shape with a working implementation behind it, so wiring it
  into the agent later would not mean redesigning storage.

### What was still weak

- The `SESSION_MEMORY` dictionary was still in use; nothing here replaced it.
- No Postgres version of either store.
- Nothing wrote facts. They could be saved by hand, but no code produced them from conversations.

---

## Phase 6 — Benchmark backed by data

`v0.6.0-data-benchmark` · 2026-05-23

### What changed

- A `--llm-judge` flag that calls Claude Haiku once per query and scores factual accuracy from 0
  to 1, recording the cost of each judgement.
- A `--benchmark` flag that writes `docs/benchmark.md` as a mode-by-metric table after any
  all-modes run.
- `agent_fixtures.json` — 12 multi-turn scenarios. They cover order lookup, refunds on delivered and
  undelivered orders, remembering an order id across turns, knowledge questions, off-topic refusals,
  and requests that need a follow-up question.
- `--agent-eval`, which runs those fixtures against the live tool loop and scores tool choice, extra
  calls, and refusals.
- `check_regression.py` and `thresholds.json` — compares a run against a saved baseline and exits 1
  if a gated metric drops more than 10%.
- `.github/workflows/eval.yml` — runs retrieval, agent, and regression checks on every pull request
  and posts the difference as a comment.

### Why it matters

- For customers: drops in answer quality or tool choice get caught before release.
- For the codebase: keyword overlap could not see a confident wrong answer, and the judge closes
  that gap. The fixture eval covers behaviour that retrieval metrics cannot reach. The CI gate turns
  a one-off benchmark into something that keeps watch.

### What was still weak

- Session memory was still held in the process and lost on restart.
- The judge scored retrieved text, not the full reply.
- `docs/benchmark.md` only fills in after a live run, so the first one had to be triggered by hand.

---

## Phase 5 — Evaluation

`v0.5.0-evaluation` · 2026-05-23

Made retrieval quality a number instead of a demo.

### What changed

- `queries.json` — 30 labeled queries across 7 categories, each naming the document that should come
  back and the words a good answer would contain.
- `run.py` — the eval runner, with per-mode and all-modes runs, precision and recall at 3, context
  relevance, latency, and estimated cost, written out as JSON.
- Two chunking strategies to compare: fixed 220-character blocks and one chunk per paragraph.
- Extra fields on every chunk (document type, source section, chunking strategy) and an index on
  category.
- `deduplicate_chunks()` drops near-identical chunks at index time.
- Voyage reranking and category pre-filtering as flags, both off by default.
- Source chips in the UI. Every reply now shows which documents backed it, with a score.
- `GET /eval` — a dashboard built from the result JSON, with a colour-coded table, latency chart, and
  per-query drill-down.

### Why it matters

- For customers: answers stopped being a black box. You can see what the bot read.
- For the codebase: retrieval could finally be compared mode against mode, and regressions became
  visible before release.

### What was still weak

- Session memory was still lost on restart.
- Answer correctness was keyword overlap only, which cannot see a fluent wrong answer.
- The eval covered retrieval, not whether the agent as a whole did the right thing.

---

## Phase 4 — Streaming and a UI

`v0.4.0-streaming-ui` · 2026-05-23

### What changed

- `POST /api/chat/stream` — server-sent events emitting tool starts, tool results, tokens, and a
  final done event as the loop runs.
- `POST /api/compare` — runs phases 1, 2 and 3 at once and returns all three.
- `frontend/index.html` — a single-file chat UI with a streaming Chat tab and a three-column Compare
  tab. No build step.
- A `mode` parameter so any phase can be tried without restarting the server.
- CORS middleware so the HTML file can reach the API.

### Why it matters

- For customers: tool activity appears as it happens. You see the order lookup and refund cards
  before the reply arrives, which makes a multi-step answer feel much faster.
- For the codebase: comparing phases became one API call, which is what made the improvement across
  releases visible in a single view.

### What was still weak

- Session memory was still lost on restart.
- Compare covered phases 1 to 3 only, since it needs responses that arrive all at once.
- There was still no eval layer. Improvements were shown by example, not measured.

---

## Phase 3 — The agent tool loop

`v0.3.0-agent-tool-loop` · 2026-05-23

### What changed

- A Claude tool loop replaced the if/else routing. The full conversation and four tool definitions
  go to the model, which runs until it is done.
- Formal schemas for `lookup_order`, `request_refund`, `search_knowledge_base` and `create_ticket`,
  with a dispatcher that runs whichever the model picks.
- Session memory now holds the full message history including tool calls and results.
- Ticket subjects are written by the model from context rather than pulled from a template.
- A refund refused on an undelivered order now offers alternatives instead of failing silently.
- The old deterministic routing stayed as a fallback, so tests still run with no API key.

### Why it matters

- For customers: "My order arrived damaged, I want a refund" now finishes in one message. "Can I get
  my money back?" turns into a normal back-and-forth instead of an immediate hand-off.
- For the codebase: routing logic no longer lives in code. Adding a tool means adding a schema and a
  function.

### What was still weak

- No streaming, so the customer waited for the whole loop before seeing anything.
- No UI, just JSON.
- Tool errors went back to the model as plain strings with no retry logic.
- Session memory was lost on restart.

---

## Phase 2 — Better retrieval

`v0.2.0-retrieval-upgrade` · 2026-05-21

### What changed

- `PostgresRepository` with full-text search and hybrid search, mixing 30% text rank with 70%
  similarity from Voyage embeddings.
- A `knowledge_chunks` table with a 512-dimension vector column and an HNSW index.
- Embedding helpers in `data_loader.py` and a `compare-retrieval` command for side-by-side checks.
- The confidence threshold moved from 0.05 to 0.25 to suit continuous scores.

### Why it matters

- For customers: questions that share no words with the knowledge base started working. "Can I get
  my money back?" reached the refund policy instead of a hand-off, and so did the two other common
  misses.
- For the codebase: retrieval quality became a continuous score rather than a count of matching
  words, and the fallback chain kept tests fast with no external services.

### What was still weak

- Routing was still if/else, with no model in the loop.
- Replies were still filled-in templates.
- The 0.25 threshold was a guess, and anything near it needed a human to check.

---

## Phase 1 — Thin vertical slice

`v0.1.0-core` · 2026-05-18

### What changed

- A Supabase schema, a repository layer, a loader for the Olist dataset, and a Postgres import path.
- Order lookup working against real imported ids instead of a handful of demo ones.
- Knowledge retrieval reading stored documents and chunks.

### Why it matters

- For customers: order questions could be answered from real data.
- For the codebase: storage and agent logic came apart, which is what made every later retrieval
  change possible without a rewrite.

### What was still weak

- Knowledge retrieval was stored, but still keyword-based.
- Refund and ticket flows were fixed rules, not model-driven.

---

## Template for the next entry

```markdown
## Phase N — Title

`vN.N.N-slug` · YYYY-MM-DD

### What changed

- Added:
- Changed:
- Removed:

### Why it matters

- For customers:
- For the codebase:

### What was still weak

-
```
