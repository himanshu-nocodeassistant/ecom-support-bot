# SupportBot

This repo includes:

- A Python backend with a minimal support agent
- Knowledge retrieval over a small in-repo knowledge base
- Deterministic order lookup and refund/ticket tool stubs
- Session-scoped memory
- A roadmap for upgrading the system release by release

## What exists today

- `backend/app/main.py`: FastAPI entrypoint
- `backend/app/agent.py`: core support-agent logic
- `backend/app/data.py`: sample orders and knowledge base
- `backend/tests/test_support_bot.py`: baseline tests
- `plans/supportbot-plan.md`: phased implementation plan
- `docs/release-roadmap.md`: GitHub release sequence
- `docs/improvement-log.md`: place to note what changed and what it unlocked

## First release goal

`v0.1.0-core` should prove the end-to-end product loop:

1. Customer asks a product or order question
2. Agent decides whether to search docs or look up an order
3. Agent returns an answer with visible tool activity
4. Agent keeps basic conversation context in-session
5. Tests pass locally

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
