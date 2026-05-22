# SupportBot

An e-commerce customer support agent built in phases to demonstrate how AI support systems work under the hood — knowledge retrieval, order lookup, session memory, escalation, and model-driven tool orchestration. Each phase is a working demo, not a toy.

**Stack:** Python, FastAPI, Claude API (Haiku), Voyage AI embeddings, Supabase/Postgres + pgvector

## What exists today (Phase 3)

### Agent
- `backend/app/agent.py` — model-driven tool loop using Claude API; four formal tool schemas; deterministic fallback when no API key is set
- `backend/app/main.py` — FastAPI entrypoint with `/health` and `/api/chat`

### Retrieval
- `backend/app/repository.py` — storage boundary; hybrid search (30% full-text + 70% cosine similarity via Voyage AI) when Postgres + API key are available; falls back to full-text, then in-memory keyword search
- `backend/knowledge/` — source markdown files for persisted support knowledge
- `backend/app/data.py` — fallback sample orders and knowledge base (no external deps needed)

### Data
- `backend/app/data_loader.py` — Olist dataset normalization and Voyage AI embedding generation
- `backend/app/cli.py` — summarize, export, import-orders, import-knowledge, compare-retrieval commands
- `backend/sql/schema.sql` — Supabase/Postgres schema (orders, knowledge_documents, knowledge_chunks + pgvector)

### Config & tests
- `backend/app/config.py` — reads `.env`; `SUPPORTBOT_DATA_BACKEND`, `DATABASE_URL`, `VOYAGE_API_KEY`, `ANTHROPIC_API_KEY`
- `backend/tests/` — unit tests (no external deps); integration tests for hybrid retrieval quality
- `plans/supportbot-plan.md` — phased implementation plan
- `docs/` — improvement log and release roadmap

## Current capabilities

| Capability | Status |
|---|---|
| Product questions from knowledge base | ✅ Hybrid semantic + keyword retrieval |
| Order status lookup | ✅ Supabase (10k Olist rows) + in-memory fallback |
| Refund flow with delivery validation | ✅ Model-driven, multi-step |
| Escalation to support ticket | ✅ Conversational, contextual ticket subject |
| Session memory across turns | ✅ Full conversation context passed to model |
| Multi-step tool chains in one message | ✅ Claude chains tools automatically |
| Confidence-based fallback | ✅ 0.25 threshold; escalates rather than guessing |

## Local run (no external deps)

Install dependencies:

```bash
pip install -r backend/requirements.txt
```

Run the API:

```bash
uvicorn backend.app.main:app --reload
```

Run tests:

```bash
python3.11 -m pytest backend/tests/ -v -k "not HybridSearch"
```

Without `.env` set, the bot uses in-memory data and deterministic routing — no API keys required.

## Full setup (Supabase + Voyage AI + Claude)

Copy and fill in `.env`:

```
SUPPORTBOT_DATA_BACKEND=postgres
DATABASE_URL=<supabase-postgres-url>
VOYAGE_API_KEY=<voyage-api-key>
ANTHROPIC_API_KEY=<anthropic-api-key>
```

Apply the schema, then import data:

```bash
# Apply schema in Supabase SQL editor
# backend/sql/schema.sql

# Import orders from Olist dataset
python3.11 -m backend.app.cli import-orders --dataset-dir ./data-set --limit 10000

# Import and embed knowledge docs
python3.11 -m backend.app.cli import-knowledge --knowledge-dir ./backend/knowledge
```

Run full test suite (hits Supabase + Voyage):

```bash
python3.11 -m pytest backend/tests/ -v
```

Compare keyword vs hybrid retrieval:

```bash
python3.11 -m backend.app.cli compare-retrieval
```

## Example requests

Product question (semantic retrieval):
```bash
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"Can I get my money back?"}' | jq
```

Multi-step refund in one message:
```bash
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"My order ORD-1002 arrived damaged, I want a refund."}' | jq
```

Real Olist order lookup:
```bash
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"Where is order 000229ec398224ef6ca0657da4fc703e?"}' | jq
```

## Verified state

- Supabase connected; schema applied
- `support_orders` — 10,000 imported Olist rows
- `knowledge_documents` — 4 imported files
- `knowledge_chunks` — 6 chunks with 512-dim Voyage embeddings + HNSW index
- Hybrid search: "Can I get my money back?" → refund policy (score 0.266); keyword search → no result
- Agentic loop: single message "arrived damaged, want refund" → `lookup_order` + `request_refund` chained automatically
