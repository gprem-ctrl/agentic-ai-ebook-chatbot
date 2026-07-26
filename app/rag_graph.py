"""
LangGraph pipeline for the RAG chatbot.

Graph shape:

    retrieve  -->  generate  -->  grade_confidence  -->  END

- retrieve: embeds the question, queries Pinecone for top-k chunks
- generate: asks the LLM (Groq/LLaMA 3) to answer strictly from those
  chunks (if none are relevant enough, skips straight to a refusal)
- grade_confidence: computes a confidence score from retrieval similarity
  and attaches it, along with the retrieved context, to the final response
"""
from typing import List, TypedDict

from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from app import config

_embedder = None
_pinecone_index = None
_llm = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _embedder


def _get_index():
    global _pinecone_index
    if _pinecone_index is None:
        from pinecone import Pinecone
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        _pinecone_index = pc.Index(config.PINECONE_INDEX_NAME)
    return _pinecone_index


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.GROQ_MODEL_NAME,
            temperature=0.1,
        )
    return _llm


class RAGState(TypedDict, total=False):
    question: str
    retrieved_chunks: List[dict]
    top_score: float
    answer: str
    confidence: float


def retrieve_node(state: RAGState) -> RAGState:
    question = state["question"]
    embedder = _get_embedder()
    query_vector = embedder.encode([question], normalize_embeddings=True)[0].tolist()

    index = _get_index()
    result = index.query(vector=query_vector, top_k=config.TOP_K, include_metadata=True)

    matches = result.get("matches", [])
    chunks = [
        {
            "text": m["metadata"]["text"],
            "page": m["metadata"].get("page"),
            "score": m["score"],
        }
        for m in matches
    ]
    top_score = chunks[0]["score"] if chunks else 0.0

    return {**state, "retrieved_chunks": chunks, "top_score": top_score}


def generate_node(state: RAGState) -> RAGState:
    question = state["question"]
    chunks = state.get("retrieved_chunks", [])
    top_score = state.get("top_score", 0.0)

    if not chunks or top_score < config.MIN_CONFIDENCE_THRESHOLD:
        return {**state, "answer": config.OUT_OF_SCOPE_MESSAGE}

    context_block = "\n\n".join(
        f"[Page {c['page']}] {c['text']}" for c in chunks
    )

    system_prompt = (
        "You are a strict, grounded assistant answering questions about the "
        "'Agentic AI: An Executive's Guide' eBook. Answer ONLY using the "
        "provided context. If the context does not contain the answer, say "
        "you don't have grounded information for that question. Do not use "
        "outside knowledge. Keep answers concise and cite page numbers "
        "in parentheses where relevant."
    )
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    return {**state, "answer": response.content.strip()}


def grade_confidence_node(state: RAGState) -> RAGState:
    top_score = state.get("top_score", 0.0)
    # top_score is already a cosine similarity in [0, 1] from Pinecone
    confidence = round(float(top_score), 4)
    return {**state, "confidence": confidence}


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("grade_confidence", grade_confidence_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "grade_confidence")
    graph.add_edge("grade_confidence", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def answer_question(question: str) -> dict:
    """Public entrypoint: run the graph and return a clean response dict."""
    graph = get_compiled_graph()
    result = graph.invoke({"question": question})
    return {
        "answer": result.get("answer", config.OUT_OF_SCOPE_MESSAGE),
        "retrieved_context": result.get("retrieved_chunks", []),
        "confidence": result.get("confidence", 0.0),
    }
