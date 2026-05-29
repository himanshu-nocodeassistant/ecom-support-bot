from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .agent import handle_compare, handle_message, handle_message_stream
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


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict:
    return handle_message(
        payload.session_id,
        payload.message,
        mode=payload.mode,
        customer_email=payload.customer_email,
    )


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
