"""
Offline unit tests: no Pinecone/Groq/HuggingFace network calls.
Run: python -m pytest tests/ -v
"""
from unittest.mock import MagicMock, patch

import numpy as np

from app import config, rag_graph
from app.ingest import chunk_text


# ---------- chunking ----------

def test_chunking_basic():
    pages = [{"page": 1, "text": "A" * 2000}]
    chunks = chunk_text(pages, chunk_size=800, overlap=120)
    assert len(chunks) == 3
    assert all(len(c["text"]) <= 800 for c in chunks)


def test_chunking_preserves_page_metadata():
    pages = [
        {"page": 1, "text": "hello world " * 100},
        {"page": 2, "text": "second page " * 100},
    ]
    chunks = chunk_text(pages, chunk_size=200, overlap=20)
    assert {c["page"] for c in chunks} == {1, 2}


def test_chunking_does_not_cut_words():
    pages = [
        {"page": 1, "text": "hello world " * 100},
        {"page": 2, "text": "second page " * 100},
    ]
    chunks = chunk_text(pages, chunk_size=200, overlap=20)
    valid = {"hello", "world", "second", "page"}
    for c in chunks:
        assert set(c["text"].split()) <= valid


def test_chunking_short_page_is_single_chunk():
    chunks = chunk_text([{"page": 1, "text": "short page text"}], chunk_size=800, overlap=120)
    assert chunks == [{"text": "short page text", "page": 1}]


# ---------- retrieve ----------

@patch("app.rag_graph._get_index")
@patch("app.rag_graph._get_embedder")
def test_retrieve_node_returns_top_score(mock_embedder, mock_index):
    mock_embedder.return_value.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_index.return_value.query.return_value = {
        "matches": [
            {"metadata": {"text": "Agentic AI is autonomous.", "page": 2.0}, "score": 0.82},
            {"metadata": {"text": "LLMs are reactive.", "page": 3.0}, "score": 0.55},
        ]
    }
    state = rag_graph.retrieve_node({"question": "What is Agentic AI?"})
    assert state["top_score"] == 0.82
    assert len(state["retrieved_chunks"]) == 2
    assert state["retrieved_chunks"][0]["page"] == 2  # float page cleaned to int
    assert isinstance(state["retrieved_chunks"][0]["page"], int)


@patch("app.rag_graph._get_index")
@patch("app.rag_graph._get_embedder")
def test_retrieve_node_handles_no_matches(mock_embedder, mock_index):
    mock_embedder.return_value.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_index.return_value.query.return_value = {"matches": []}
    state = rag_graph.retrieve_node({"question": "anything"})
    assert state["retrieved_chunks"] == []
    assert state["top_score"] == 0.0


# ---------- routing ----------

def test_route_refuses_when_no_chunks():
    assert rag_graph.route_after_retrieve({"retrieved_chunks": [], "top_score": 0.0}) == "refuse"


def test_route_refuses_below_threshold():
    state = {
        "retrieved_chunks": [{"text": "x", "page": 1, "score": 0.1}],
        "top_score": config.MIN_CONFIDENCE_THRESHOLD - 0.01,
    }
    assert rag_graph.route_after_retrieve(state) == "refuse"


def test_route_generates_at_threshold():
    state = {
        "retrieved_chunks": [{"text": "x", "page": 1, "score": 0.5}],
        "top_score": config.MIN_CONFIDENCE_THRESHOLD,
    }
    assert rag_graph.route_after_retrieve(state) == "generate"


def test_refuse_node_sets_out_of_scope_message():
    result = rag_graph.refuse_node({"question": "capital of France?"})
    assert result["answer"] == config.OUT_OF_SCOPE_MESSAGE
    assert result["refused"] is True


# ---------- generate ----------

@patch("app.rag_graph._get_llm")
def test_generate_node_returns_llm_answer(mock_llm):
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
    assert result["refused"] is False
    mock_llm.return_value.invoke.assert_called_once()


@patch("app.rag_graph._get_llm")
def test_generate_node_maps_llm_refusal_to_out_of_scope(mock_llm):
    mock_response = MagicMock()
    mock_response.content = rag_graph.NOT_IN_CONTEXT
    mock_llm.return_value.invoke.return_value = mock_response

    state = {
        "question": "Something the context doesn't cover",
        "retrieved_chunks": [{"text": "Unrelated text.", "page": 5, "score": 0.4}],
        "top_score": 0.4,
    }
    result = rag_graph.generate_node(state)
    assert result["answer"] == config.OUT_OF_SCOPE_MESSAGE
    assert result["refused"] is True


# ---------- confidence ----------

def test_grade_confidence_node():
    result = rag_graph.grade_confidence_node({"top_score": 0.7321, "refused": False})
    assert result["confidence"] == 0.7321


def test_grade_confidence_is_zero_when_refused():
    result = rag_graph.grade_confidence_node({"top_score": 0.6, "refused": True})
    assert result["confidence"] == 0.0


# ---------- full graph ----------

@patch("app.rag_graph._get_llm")
@patch("app.rag_graph._get_index")
@patch("app.rag_graph._get_embedder")
def test_full_graph_in_scope(mock_embedder, mock_index, mock_llm):
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


@patch("app.rag_graph._get_llm")
@patch("app.rag_graph._get_index")
@patch("app.rag_graph._get_embedder")
def test_full_graph_out_of_scope_skips_llm(mock_embedder, mock_index, mock_llm):
    mock_embedder.return_value.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_index.return_value.query.return_value = {
        "matches": [{"metadata": {"text": "Unrelated.", "page": 1}, "score": 0.05}]
    }

    result = rag_graph.answer_question("What is the capital of France?")
    assert result["answer"] == config.OUT_OF_SCOPE_MESSAGE
    assert result["confidence"] == 0.0
    mock_llm.assert_not_called()


@patch("app.rag_graph._get_llm")
@patch("app.rag_graph._get_index")
@patch("app.rag_graph._get_embedder")
def test_full_graph_llm_refusal_gives_zero_confidence(mock_embedder, mock_index, mock_llm):
    mock_embedder.return_value.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_index.return_value.query.return_value = {
        "matches": [{"metadata": {"text": "Loosely related.", "page": 4}, "score": 0.45}]
    }
    mock_response = MagicMock()
    mock_response.content = rag_graph.NOT_IN_CONTEXT
    mock_llm.return_value.invoke.return_value = mock_response

    result = rag_graph.answer_question("A question the eBook doesn't answer")
    assert result["answer"] == config.OUT_OF_SCOPE_MESSAGE
    assert result["confidence"] == 0.0
