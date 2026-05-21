# SupportBot

This repo includes:

- A Python backend with a minimal support agent
- Knowledge retrieval over a small in-repo knowledge base
- Deterministic order lookup and refund/ticket tool stubs
- Session-scoped memory
- A repository layer that can switch between in-memory data and Postgres-backed data
- A Supabase-backed order store seeded from the Olist dataset
- An Olist dataset loader, knowledge import CLI, and SQL schema for Supabase
- A roadmap for upgrading the system release by release

## What exists today

- `backend/app/main.py`: FastAPI entrypoint
- `backend/app/agent.py`: core support-agent logic
- `backend/app/repository.py`: storage boundary for in-memory and Postgres-backed reads
- `backend/app/data_loader.py`: Olist normalization and import logic
- `backend/knowledge/`: source markdown files for persisted support knowledge
- `backend/app/cli.py`: dataset summarize/export/import commands
- `backend/sql/schema.sql`: Supabase/Postgres schema
- `backend/app/data.py`: fallback sample orders and knowledge base
- `backend/tests/test_support_bot.py`: baseline tests
- `backend/tests/test_data_loader.py`: dataset loader coverage
- `plans/supportbot-plan.md`: phased implementation plan
- `docs/release-roadmap.md`: GitHub release sequence
- `docs/improvement-log.md`: place to note what changed and what it unlocked

Current scope:

- No vector database
- No hybrid retrieval
- No Claude tool loop
- No streaming
- No Helicone
- No RAG showcase UI

## Local run

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
python3 -m unittest discover -s backend/tests
```

## Data setup

The app now supports real order lookups against Supabase through `.env`:

1. Set `SUPPORTBOT_DATA_BACKEND=postgres`
2. Set `DATABASE_URL` to your Supabase Postgres connection string
3. Set `OLIST_DATASET_DIR` to the local Olist CSV directory
4. Apply [schema.sql](/Users/himanshusharma/Code/Codex/ecom-support-bot/backend/sql/schema.sql)
5. Import orders with the CLI

Current verified state:

- Supabase connection works
- Schema is applied
- `support_orders` contains `10,000` imported Olist rows
- `knowledge_documents` contains `4` imported files
- `knowledge_chunks` contains `6` imported chunks
- The agent can resolve real 32-character Olist order IDs
- Product questions can be answered from persisted knowledge chunks in Supabase

Example commands:

```bash
python3 -m backend.app.cli summarize-olist --dataset-dir ./data-set --limit 5
python3 -m backend.app.cli export-orders --dataset-dir ./data-set --output ./local/normalized_orders.csv --limit 1000
python3 -m backend.app.cli import-orders --dataset-dir ./data-set --limit 1000
python3 -m backend.app.cli import-knowledge --knowledge-dir ./backend/knowledge
```

Example live lookup:

```bash
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-real","message":"Where is my order 000229ec398224ef6ca0657da4fc703e?"}'
```
