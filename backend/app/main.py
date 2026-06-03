from __future__ import annotations

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .agent import SESSION_MEMORY, handle_compare, handle_message, handle_message_stream
from .eval_dashboard import render_dashboard

app = FastAPI(title="SupportBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    mode: str = "phase3"
    customer_email: str | None = None


class CompareRequest(BaseModel):
    message: str


class StreamRequest(BaseModel):
    session_id: str
    message: str
    customer_email: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def run_fact_extraction(session_id: str, customer_email: str) -> None:
    """Load conversation from SESSION_MEMORY, extract facts, persist to customer store."""
    from .agent import _default_customer_store
    from .config import get_settings
    from .fact_extractor import extract_facts_from_conversation

    settings = get_settings()
    raw_messages = SESSION_MEMORY.get(session_id, [])
    conversation = [
        {"role": m["role"], "content": m["content"]}
        for m in raw_messages
        if isinstance(m.get("content"), str)
    ]
    if not conversation:
        return
    facts = extract_facts_from_conversation(conversation, api_key=settings.anthropic_api_key)
    if not facts:
        return
    customer = _default_customer_store.upsert_customer(email=customer_email, name="")
    for fact in facts:
        _default_customer_store.save_memory_fact(
            customer_id=customer["customer_id"],
            fact_type=fact["fact_type"],
            fact_text=fact["fact_text"],
            confidence=fact["confidence"],
            source_session_id=session_id,
        )


@app.post("/api/chat")
async def chat(payload: ChatRequest, background_tasks: BackgroundTasks) -> dict:
    result = handle_message(
        payload.session_id,
        payload.message,
        mode=payload.mode,
        customer_email=payload.customer_email,
    )
    if payload.customer_email:
        background_tasks.add_task(
            run_fact_extraction,
            session_id=payload.session_id,
            customer_email=payload.customer_email,
        )
    return result


@app.post("/api/chat/stream")
async def chat_stream(payload: StreamRequest):
    async def event_generator():
        async for chunk in handle_message_stream(
            payload.session_id,
            payload.message,
            customer_email=payload.customer_email,
        ):
            yield chunk

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/compare")
async def compare(payload: CompareRequest) -> dict:
    return await handle_compare(payload.message)


@app.get("/eval", response_class=HTMLResponse)
def eval_dashboard() -> str:
    return render_dashboard()


@app.get("/eval/memory")
def eval_memory() -> dict:
    import json
    from pathlib import Path

    from backend.eval.memory_eval import run_memory_eval

    fixtures_path = Path(__file__).parent.parent / "eval" / "memory_fixtures.json"
    fixtures = json.loads(fixtures_path.read_text()) if fixtures_path.exists() else []
    result = run_memory_eval(fixtures)

    results_dir = Path(__file__).parent.parent / "eval" / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "memory_eval.json").write_text(json.dumps(result, indent=2))

    return result
