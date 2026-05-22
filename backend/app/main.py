from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .agent import handle_compare, handle_message, handle_message_stream

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


class CompareRequest(BaseModel):
    message: str


class StreamRequest(BaseModel):
    session_id: str
    message: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict:
    return handle_message(payload.session_id, payload.message, mode=payload.mode)


@app.post("/api/chat/stream")
async def chat_stream(payload: StreamRequest):
    async def event_generator():
        async for chunk in handle_message_stream(payload.session_id, payload.message):
            yield chunk

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/compare")
async def compare(payload: CompareRequest) -> dict:
    return await handle_compare(payload.message)
