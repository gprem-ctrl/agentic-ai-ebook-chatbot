"""
Optional Streamlit UI for the RAG chatbot (the task allows either an API
or a chat UI — this app talks to the same rag_graph pipeline directly,
so it works with or without the FastAPI server running).

Run:
    streamlit run app/ui.py
"""
import streamlit as st
from app.rag_graph import answer_question

st.set_page_config(page_title="Agentic AI eBook Chatbot", page_icon="🤖")
st.title("🤖 Agentic AI eBook Chatbot")
st.caption("RAG chatbot grounded strictly in the 'Agentic AI: An Executive's Guide' eBook "
           "— built with LangGraph + Pinecone.")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask a question about the Agentic AI eBook...")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        st.caption(f"Confidence: {turn['confidence']:.2f}")
        with st.expander("Retrieved context"):
            for chunk in turn["retrieved_context"]:
                st.markdown(f"**Page {chunk.get('page')}** (score: {chunk.get('score', 0):.3f})")
                st.write(chunk["text"])

if question:
    with st.spinner("Retrieving and generating..."):
        result = answer_question(question)
    st.session_state.history.append({"question": question, **result})
    st.rerun()
