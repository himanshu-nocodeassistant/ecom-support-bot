# Plan: Phase 4 — Streaming Product Experience

> Source PRD: Phase 4 of supportbot-plan.md — Streaming Product Experience with Replay/Diff Showcase UI

## Architectural decisions

- **Execution modes**: four named modes — `phase1` (keyword), `phase2` (hybrid), `phase3` (agent loop), `phase4` (streaming agent loop). Mode is a request param, not a server config.
- **Compare endpoint**: `POST /api/compare` runs phase1, phase2, phase3 in parallel and returns a single JSON object with all three results. No streaming on the compare path.
- **Streaming transport**: SSE (`text/event-stream`) on `GET /api/chat/stream`. Event types: `token`, `tool_start`, `tool_result`, `done`, `error`.
- **Repository routing**: mode determines which repository method is called — `InMemoryRepository.search_knowledge` for phase1, `PostgresRepository._fulltext_search` for phase2 (FTS only, no embeddings), `PostgresRepository._hybrid_search` for phase2 with Voyage key. Existing `get_repository()` factory is unchanged; mode-aware helpers wrap it.
- **Frontend**: single `frontend/index.html` file — no build tooling, no bundler. Vanilla JS + CSS. Two tabs: Chat (SSE stream, phase selector) and Compare (diff cards).
- **Session memory**: unchanged — in-process dict, scoped per session_id. Compare mode uses a throwaway session ID per request.
- **CORS**: FastAPI CORS middleware added to allow the HTML file to call the API when opened from file:// or a different port.

---

## Phase 4.1: Phase-aware backend + `/api/compare`

### What to build

Extend `agent.py` with a `mode` parameter that selects the execution path:
- `phase1`: deterministic routing (existing `_handle_message_deterministic`) forced onto `InMemoryRepository` regardless of env config
- `phase2`: deterministic routing forced onto `PostgresRepository` FTS (or hybrid if Voyage key present)
- `phase3`: Claude agent tool loop (existing `handle_message`) — no change in behavior
- `phase4`: same as phase3 for now (streaming added in 4.2)

Add `POST /api/compare` to `main.py` that calls all three non-streaming modes concurrently (asyncio `gather`) and returns `{phase1: {...}, phase2: {...}, phase3: {...}}`.

Update `POST /api/chat` to accept an optional `mode` field (default `phase3`).

This phase is fully testable via curl with no frontend.

### Acceptance criteria

- [ ] `POST /api/chat` with `mode=phase1` uses keyword-only retrieval regardless of `SUPPORTBOT_DATA_BACKEND`
- [ ] `POST /api/chat` with `mode=phase2` uses FTS or hybrid retrieval
- [ ] `POST /api/chat` with `mode=phase3` behaves identically to the current default
- [ ] `POST /api/compare` returns results for all three phases in one response
- [ ] Existing tests still pass

---

## Phase 4.2: SSE streaming endpoint

### What to build

Add an async generator `handle_message_stream` to `agent.py` that wraps the Anthropic streaming SDK (`client.messages.stream()`). It yields SSE-formatted events:
- `tool_start` — emitted when Claude calls a tool, before execution
- `tool_result` — emitted after the tool returns
- `token` — each text delta from the model
- `done` — final event, carries `session_id` and total tool event count
- `error` — on exception

Add `POST /api/chat/stream` to `main.py` returning `StreamingResponse(media_type="text/event-stream")`.

Phase4 mode on `/api/chat` routes to the streaming path.

### Acceptance criteria

- [ ] `curl -N POST /api/chat/stream` shows tokens arriving progressively
- [ ] Tool events appear in the stream before the model's next reply token
- [ ] First token arrives within 500ms for a single-tool query (measured manually)
- [ ] Stream closes cleanly with a `done` event on normal completion
- [ ] Stream emits `error` event and closes on exception rather than hanging

---

## Phase 4.3: Chat UI

### What to build

`frontend/index.html` — a single-file chat interface with:
- Phase selector tabs (Phase 1 / 2 / 3 / 4) that set the active mode
- Message thread: user bubbles on the right, assistant on the left
- Tool activity shown as inline cards between the last user message and the final reply — each card names the tool and shows collapsed input/output
- Streaming: Phase 4 tab connects to `/api/chat/stream` and renders tokens as they arrive; other tabs hit `POST /api/chat` with the appropriate mode and render on completion
- Loading state while waiting for first token
- Error state if the API returns an error or the stream fails
- "Compare →" button in the nav that switches to the compare tab

No framework, no build step. The file opens directly in a browser pointed at `http://localhost:8000`.

### Acceptance criteria

- [ ] Phase 4 tab streams tokens visibly — text appears word by word
- [ ] Tool events are visible as inline cards before the final reply
- [ ] Switching phase tabs changes which backend mode is called
- [ ] Loading and error states are handled gracefully
- [ ] The UI is usable on a 1280px screen without horizontal scroll

---

## Phase 4.4: Compare / diff view

### What to build

Add a second tab "Compare" to `frontend/index.html`:
- Single query input + submit button
- On submit: calls `POST /api/compare`, shows a spinner
- On response: renders three cards side by side — Phase 1, Phase 2, Phase 3
- Each card shows:
  - Phase label and capability badge (e.g. "Keyword retrieval", "Hybrid RAG", "Agent loop")
  - The reply text
  - Tool events list (name + collapsed input/output), or "No tools used"
  - Retrieval score badge if `search_knowledge_base` was called (pulled from tool output)
- Cards highlight the key difference: Phase 1 escalates where Phase 2 returns a scored match; Phase 3 chains tools
- A set of 3–4 pre-written "demo queries" as clickable chips (e.g. "Can I get a refund?", "Where is my order ORD-1001?", "Does the portable blender have a safety lock?") so reviewers can see the evolution without typing

### Acceptance criteria

- [ ] Submitting a query renders all three phase cards simultaneously
- [ ] Retrieval score badge appears on Phase 1 and Phase 2 cards when knowledge base was searched
- [ ] Tool event list is accurate per phase (Phase 1/2 show deterministic tool calls, Phase 3 shows model-driven ones)
- [ ] Demo query chips pre-fill the input and auto-submit
- [ ] Cards are readable side by side on a 1280px screen
- [ ] The page feels like a coherent product showcase, not a raw API debug view
