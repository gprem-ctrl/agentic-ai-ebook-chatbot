"""
FastAPI service exposing the RAG chatbot.

Run:
    uvicorn app.api:app --reload --port 8000

Endpoint:
    POST /chat
    body: {"question": "What is Agentic AI?"}
    returns: {"answer": ..., "retrieved_context": [...], "confidence": ...}
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from app.rag_graph import answer_question

app = FastAPI(
    title="Agentic AI eBook RAG Chatbot",
    description="RAG chatbot answering questions strictly from the Agentic AI eBook, "
                "built with LangGraph + Pinecone.",
    version="1.0.0",
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
        result = answer_question(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {e}")
    return result
