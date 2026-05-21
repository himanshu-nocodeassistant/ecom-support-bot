from fastapi import FastAPI
from pydantic import BaseModel

from .agent import handle_message

app = FastAPI(title="SupportBot API")


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict:
    return handle_message(payload.session_id, payload.message)
