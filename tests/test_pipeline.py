"""
Offline unit tests — no Pinecone/Groq/HuggingFace network calls.
Run: python -m pytest tests/ -v
"""
import numpy as np
from unittest.mock import MagicMock, patch
from app.ingest import chunk_text
from app import rag_graph, config


def test_chunking_basic():
    pages = [{"page": 1, "text": "A" * 2000}]
    chunks = chunk_text(pages, chunk_size=800, overlap=120)
    assert len(chunks) == 3
    assert all(len(c["text"]) <= 800 for c in chunks)


def test_chunking_preserves_page_metadata():
    pages = [{"page": 1, "text": "hello world " * 100}, {"page": 2, "text": "second page " * 100}]
    chunks = chunk_text(pages, chunk_size=200, overlap=20)
    pages_seen = {c["page"] for c in chunks}
    assert pages_seen == {1, 2}


@patch("app.rag_graph._get_index")
@patch("app.rag_graph._get_embedder")
def test_retrieve_node_returns_top_score(mock_embedder, mock_index):
    mock_embedder.return_value.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_index.return_value.query.return_value = {
        "matches": [
            {"metadata": {"text": "Agentic AI is autonomous.", "page": 2}, "score": 0.82},
            {"metadata": {"text": "LLMs are reactive.", "page": 3}, "score": 0.55},
        ]
    }
    state = rag_graph.retrieve_node({"question": "What is Agentic AI?"})
    assert state["top_score"] == 0.82
    assert len(state["retrieved_chunks"]) == 2


@patch("app.rag_graph._get_llm")
def test_generate_node_refuses_below_threshold(mock_llm):
    state = {
        "question": "What is the capital of France?",
        "retrieved_chunks": [{"text": "irrelevant", "page": 1, "score": 0.1}],
        "top_score": 0.1,
    }
    result = rag_graph.generate_node(state)
    assert result["answer"] == config.OUT_OF_SCOPE_MESSAGE
    mock_llm.assert_not_called()  # should short-circuit, never call the LLM


@patch("app.rag_graph._get_llm")
def test_generate_node_calls_llm_above_threshold(mock_llm):
    mock_response = MagicMock()
    mock_response.content = "Agentic AI is autonomous and goal-driven (Page 2)."
    mock_llm.return_value.invoke.return_value = mock_response

    state = {
        "question": "What is Agentic AI?",
        "retrieved_chunks": [{"text": "Agentic AI is autonomous.", "page": 2, "score": 0.82}],
        "top_score": 0.82,
    }
    result = rag_graph.generate_node(state)
    assert "autonomous" in result["answer"]
    mock_llm.return_value.invoke.assert_called_once()


def test_grade_confidence_node():
    state = {"top_score": 0.7321}
    result = rag_graph.grade_confidence_node(state)
    assert result["confidence"] == 0.7321


@patch("app.rag_graph._get_llm")
@patch("app.rag_graph._get_index")
@patch("app.rag_graph._get_embedder")
def test_full_graph_end_to_end(mock_embedder, mock_index, mock_llm):
    mock_embedder.return_value.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_index.return_value.query.return_value = {
        "matches": [{"metadata": {"text": "Agentic AI is autonomous.", "page": 2}, "score": 0.9}]
    }
    mock_response = MagicMock()
    mock_response.content = "Agentic AI is an autonomous, goal-driven system (Page 2)."
    mock_llm.return_value.invoke.return_value = mock_response

    result = rag_graph.answer_question("What is Agentic AI?")
    assert result["confidence"] == 0.9
    assert len(result["retrieved_context"]) == 1
    assert "autonomous" in result["answer"]
