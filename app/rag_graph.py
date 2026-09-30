"""
LangGraph pipeline for the RAG chatbot.

Graph shape:

    retrieve --(top score >= threshold)--> generate --> grade_confidence --> END
    retrieve --(top score <  threshold)--> refuse   --> grade_confidence --> END

- retrieve: embeds the question and queries Pinecone for the top-k chunks
- generate: asks the LLM (Groq / LLaMA 3) to answer strictly from those chunks;
  if the LLM says the context doesn't contain the answer, the bot refuses
- refuse: returns the out-of-scope message without calling the LLM
- grade_confidence: confidence = top retrieval cosine similarity (0.0 if refused)
"""
from typing import List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app import config

NOT_IN_CONTEXT = "NOT_IN_CONTEXT"

_embedder = None
_pinecone_index = None
_llm = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
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
        from langchain_groq import ChatGroq
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
    refused: bool
    confidence: float


def retrieve_node(state: RAGState) -> RAGState:
    question = state["question"]
    query_vector = _get_embedder().encode([question], normalize_embeddings=True)[0].tolist()

    result = _get_index().query(vector=query_vector, top_k=config.TOP_K, include_metadata=True)

    chunks = []
    for m in result.get("matches", []):
        page = m["metadata"].get("page")
        chunks.append({
            "text": m["metadata"]["text"],
            # Pinecone returns numeric metadata as floats (2.0); show clean ints.
            "page": int(page) if page is not None else None,
            "score": m["score"],
        })
    top_score = chunks[0]["score"] if chunks else 0.0

    return {**state, "retrieved_chunks": chunks, "top_score": top_score}


def route_after_retrieve(state: RAGState) -> str:
    if not state.get("retrieved_chunks") or state.get("top_score", 0.0) < config.MIN_CONFIDENCE_THRESHOLD:
        return "refuse"
    return "generate"


def refuse_node(state: RAGState) -> RAGState:
    return {**state, "answer": config.OUT_OF_SCOPE_MESSAGE, "refused": True}


def generate_node(state: RAGState) -> RAGState:
    question = state["question"]
    chunks = state["retrieved_chunks"]

    context_block = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in chunks)

    system_prompt = (
        "You are a strict, grounded assistant answering questions about the "
        "Agentic AI eBook. Answer ONLY using the provided context. "
        f"If the context does not contain the answer, reply with exactly: {NOT_IN_CONTEXT}. "
        "Do not use outside knowledge. Keep answers concise and cite page "
        "numbers in parentheses where relevant."
    )
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"

    response = _get_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    answer = response.content.strip()

    if NOT_IN_CONTEXT in answer:
        return {**state, "answer": config.OUT_OF_SCOPE_MESSAGE, "refused": True}
    return {**state, "answer": answer, "refused": False}


def grade_confidence_node(state: RAGState) -> RAGState:
    if state.get("refused"):
        confidence = 0.0
    else:
        # top_score is already a cosine similarity from Pinecone
        confidence = round(float(state.get("top_score", 0.0)), 4)
    return {**state, "confidence": confidence}


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("grade_confidence", grade_confidence_node)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"generate": "generate", "refuse": "refuse"},
    )
    graph.add_edge("generate", "grade_confidence")
    graph.add_edge("refuse", "grade_confidence")
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
    result = get_compiled_graph().invoke({"question": question})
    return {
        "answer": result.get("answer", config.OUT_OF_SCOPE_MESSAGE),
        "retrieved_context": result.get("retrieved_chunks", []),
        "confidence": result.get("confidence", 0.0),
    }
