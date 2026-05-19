from fastapi import FastAPI

from app.agent import SHLAgent
from app.schemas import ChatRequest, ChatResponse


app = FastAPI(
    title="Conversational SHL Assessment Recommender",
    version="1.0.0",
)
agent = SHLAgent()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return agent.chat(request.messages)
