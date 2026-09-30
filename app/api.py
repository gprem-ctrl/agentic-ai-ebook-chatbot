"""
FastAPI service exposing the RAG chatbot.

Run:
    uvicorn app.api:app --reload --port 8000

Endpoint:
    POST /chat
    body:    {"question": "What is Agentic AI?"}
    returns: {"answer": ..., "retrieved_context": [...], "confidence": ...}
"""
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import config, rag_graph

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = [k for k in ("PINECONE_API_KEY", "GROQ_API_KEY") if not getattr(config, k)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}. See .env.example.")
    rag_graph._get_embedder()  # warm up so the first request isn't slow
    yield


app = FastAPI(
    title="Agentic AI eBook RAG Chatbot",
    description="RAG chatbot answering questions strictly from the Agentic AI eBook, "
                "built with LangGraph + Pinecone.",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question")


class ContextChunk(BaseModel):
    text: str
    page: Optional[int] = None
    score: float


class ChatResponse(BaseModel):
    answer: str
    retrieved_context: List[ContextChunk]
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        return rag_graph.answer_question(request.question)
    except Exception:
        logger.exception("RAG pipeline error")
        raise HTTPException(status_code=500, detail="Internal error while answering the question.")
