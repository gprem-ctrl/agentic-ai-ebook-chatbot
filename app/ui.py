"""
Optional Streamlit UI for the RAG chatbot. It calls the same rag_graph
pipeline directly, so it works with or without the FastAPI server running.

Run from the repo root:
    streamlit run app/ui.py
"""
import logging
import os
import sys

# Make `from app...` imports work when launched via `streamlit run app/ui.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app import config
from app.rag_graph import answer_question

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Agentic AI eBook Chatbot", page_icon="🤖")
st.title("🤖 Agentic AI eBook Chatbot")
st.caption("RAG chatbot grounded strictly in the Agentic AI eBook, "
           "built with LangGraph + Pinecone.")

missing = [k for k in ("PINECONE_API_KEY", "GROQ_API_KEY") if not getattr(config, k)]
if missing:
    st.error(f"Missing environment variables: {', '.join(missing)}. See .env.example.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []


def render_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["confidence"] > 0:
            st.caption(f"Confidence: {turn['confidence']:.2f}")
        else:
            st.caption("Not covered by the eBook")
        with st.expander("Retrieved context"):
            if not turn["retrieved_context"]:
                st.write("No chunks retrieved.")
            for chunk in turn["retrieved_context"]:
                st.markdown(f"**Page {chunk.get('page')}** (score: {chunk.get('score', 0):.3f})")
                st.write(chunk["text"])


for turn in st.session_state.history:
    render_turn(turn)

question = st.chat_input("Ask a question about the Agentic AI eBook...")

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                result = answer_question(question)
            except Exception:
                logger.exception("RAG pipeline error")
                st.error("Something went wrong while answering. Check your API keys, "
                         "that the Pinecone index exists (run `python -m app.ingest`), "
                         "and your network connection.")
                st.stop()
    st.session_state.history.append({"question": question, **result})
    st.rerun()
