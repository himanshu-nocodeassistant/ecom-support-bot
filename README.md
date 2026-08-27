# SupportBot

SupportBot is an e-commerce support bot that answers policy questions, looks up orders, requests eligible refunds, creates support tickets, and remembers customer context across sessions.

[Read the build notes](https://himanshu-sharma.medium.com/i-kept-adding-ai-to-my-support-bot-it-kept-breaking-in-new-ways-49b1ae4ff658)

Stack: Python, FastAPI, Claude API, Voyage AI, Supabase, Postgres, and pgvector.

## Measured decisions

The current evaluation set has 15 documents and 62 labelled queries. It is used to compare retrieval changes and stop regressions before release.

| Decision | Result | What changed |
|---|---|---|
| Semantic or fixed chunking | hit_rate@3 was 0.327 for semantic and 0.346 for fixed | The earlier 29% claim was removed because the metric favoured smaller chunks. No config changed. ([decision](plans/decisions/chunking.md)) |
| Voyage reranking | NDCG@5 went from 0.934 to 0.960. H@1 went from 0.904 to 0.942 | Reranking stays optional because it adds about 326 ms and increases Voyage cost per query. ([decision](plans/decisions/reranking.md)) |
| Category filter | It caused a 4.3% hard miss rate on multi-intent queries | The filter was removed. |
| Keyword and hybrid baseline | Title matching penalised chunked results | Metrics now compare document IDs. ([investigation](plans/decisions/retrieval-finding.md)) |
| Confidence threshold | The highest off-topic score was 0.294. The lowest relevant score was 0.432 | The shared threshold is 0.30. |
| CI retrieval gate | A gated metric can fall by up to 10% from the saved baseline | Larger drops fail CI. |

## Repository guide

- [`plans/decisions/`](plans/decisions/) contains the measured architecture decisions.
- [`docs/changelog.md`](docs/changelog.md) contains the phase history.
- [`docs/benchmark.md`](docs/benchmark.md) contains benchmark results.
- [`docs/eval.md`](docs/eval.md) defines the metrics and evaluation limits.
- [`plans/roadmap.md`](plans/roadmap.md) lists remaining work.
- [`plans/archive/`](plans/archive/) contains old plans.

## Request flow

Start the server, open [`frontend/index.html`](frontend/index.html), and send:

> My order ORD-1002 arrived damaged, I want a refund.

The request shows:

- streamed response tokens.
- `lookup_order` and `request_refund` tool activity.
- the knowledge documents used for the answer.
- phase 1, 2, and 3 responses on the same query.

## Current features

| Feature | Status |
|---|---|
| Knowledge questions | Hybrid text and vector retrieval with optional Voyage reranking |
| Order lookup | Supabase with an in-memory fallback |
| Refund requests | Checks delivery status before the request |
| Support tickets | Creates a ticket from the conversation |
| Chat UI | Streams tokens, tool activity, and sources over SSE |
| Customer memory | Stores orders and facts in Postgres with a 90-day fact TTL |
| Evaluation | Retrieval, agent, adversarial, synthetic, and end-to-end checks |
| Evaluation dashboard | Available at `GET /eval` and `GET /eval/memory` |
| CI | Checks saved retrieval and adversarial baselines |
| Tracing | Optional Langfuse traces for live evaluation |

## Known gaps

Before production use, add authentication, rate limiting, connection pooling, and an iteration limit for the tool loop. CORS is open. Settings are read from `.env` on each request. Customer memory is added to the system prompt without sanitisation. Raw errors can reach clients.

[`plans/roadmap.md`](plans/roadmap.md) tracks this work.

## Local run without external services

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload
```

Run tests:

```bash
python3 -m pytest backend/tests/ -v --tb=short
```

Integration tests skip when `DATABASE_URL` or `VOYAGE_API_KEY` is absent. Without `.env`, the bot uses in-memory data and deterministic routing.

## Full setup

Add these values to `.env`:

```text
SUPPORTBOT_DATA_BACKEND=postgres
DATABASE_URL=<supabase-postgres-url>
VOYAGE_API_KEY=<voyage-api-key>
ANTHROPIC_API_KEY=<anthropic-api-key>
```

Apply [`backend/sql/schema.sql`](backend/sql/schema.sql) in Supabase. Then apply these migrations:

- `migrate_5c_metadata.sql`
- `migrate_7_customer_memory.sql`
- `migrate_8_memory_wiring.sql`

Import the included Olist data and the knowledge documents:

```bash
python3 -m backend.app.cli import-orders --dataset-dir ./data-set --limit 10000
python3 -m backend.app.cli import-knowledge --knowledge-dir ./backend/knowledge
```

## Example requests

Ask a policy question:

```bash
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"example","message":"Can I get my money back?"}' | jq
```

Request a refund:

```bash
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"example","message":"My order ORD-1002 arrived damaged, I want a refund."}' | jq
```

Use customer memory:

```bash
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"example","message":"What is my order status?","customer_email":"alice@example.com"}' | jq
```

## Evaluation

```bash
# Run retrieval evaluation for all modes and update the benchmark.
python -m backend.eval.run --all-modes --benchmark

# Add LLM judgement of retrieved context.
python -m backend.eval.run --all-modes --llm-judge --benchmark

# Run agent, adversarial, and end-to-end evaluations.
python -m backend.eval.run --agent-eval
python -m backend.eval.run --adversarial-eval
python -m backend.eval.run --e2e-eval

# Generate synthetic queries and evaluate them.
python -m backend.eval.generate_queries --api-key $ANTHROPIC_API_KEY
python -m backend.eval.run --all-modes --query-set synthetic
python -m backend.eval.run --all-modes --query-set both
```

Results are written to `backend/eval/results/`. The browser dashboard is at `GET /eval` while the API is running.

The evaluation covers retrieval, tool selection, refusals, adversarial queries, and a small end-to-end set. It doesn't cover every order ID format, every unknown order case, or all differences between the memory and Postgres backends.

## A retrieval metric was wrong

An older evaluation made keyword search look better than hybrid search. The metric compared chunk titles instead of document IDs, so it penalised modes that returned several chunks from the correct document. The category filter was also removed after it caused a 4.3% hard miss rate on multi-intent queries.

The full notes are in [`plans/decisions/retrieval-finding.md`](plans/decisions/retrieval-finding.md).

Built by [Himanshu Sharma](https://nocodeassistant.agency).
