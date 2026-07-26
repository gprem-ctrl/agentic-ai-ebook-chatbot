# Agentic AI eBook RAG Chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions **strictly**
from the *"Agentic AI: An Executive's Guide"* eBook (Konverge AI / Emergence AI),
built with **LangGraph**, **Pinecone**, and local sentence embeddings.

Built for the Appening Infotech AI Engineer Intern interview task.

## Features

- Downloads and ingests the source PDF, chunks it, embeds it, and stores it in Pinecone
- LangGraph pipeline: `retrieve → generate → grade_confidence`
- Answers are grounded only in retrieved chunks — if nothing relevant is found,
  the bot says so instead of guessing
- Every response returns: **final answer**, **retrieved context chunks**, and a
  **confidence score** (cosine similarity of the top match)
- Exposed as both a **FastAPI** JSON API and an optional **Streamlit** chat UI

## Architecture

```
                     ┌─────────────────────┐
                     │   Ebook-Agentic-AI  │
                     │        .pdf         │
                     └──────────┬──────────┘
                                │ download + extract (pypdf)
                                ▼
                     ┌─────────────────────┐
                     │   Chunker (800/120  │
                     │   char sliding win) │
                     └──────────┬──────────┘
                                │ embed (all-MiniLM-L6-v2, local)
                                ▼
                     ┌─────────────────────┐
                     │  Pinecone Index     │
                     │  (cosine, 384-dim)  │
                     └──────────┬──────────┘
                                │
      ┌─────────────────────────┼─────────────────────────┐
      │                LangGraph pipeline                 │
      │                                                    │
      │   ┌──────────┐    ┌──────────┐    ┌─────────────┐  │
      │   │ retrieve │───▶│ generate │───▶│grade_confid.│  │
      │   └──────────┘    └──────────┘    └─────────────┘  │
      │   embed query      Groq LLaMA 3      similarity     │
      │   query Pinecone    grounded answer    score        │
      └────────────────────────────────────────────────────┘
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
           FastAPI  /chat                Streamlit UI
     {answer, context, confidence}      (chat_input loop)
```

**Why this stack:**
- **Embeddings run locally** (`sentence-transformers`) — no paid API key needed
  just to embed text, and results are fully reproducible.
- **Pinecone** is the vector DB, as requested in the task (serverless free tier
  is enough for this ~50-chunk document).
- **Groq + LLaMA 3** for generation — fast, free-tier friendly, and grounded via
  a strict system prompt that forbids answering outside the retrieved context.
- **Confidence score** is the top Pinecone cosine similarity score. Below a
  configurable threshold (`MIN_CONFIDENCE_THRESHOLD`, default `0.35`), the graph
  skips the LLM call entirely and returns a refusal — this keeps the bot honest
  and saves an LLM call on clearly out-of-scope questions.

## Setup

### 1. Clone and install

```bash
git clone <this-repo-url>
cd agentic-rag-chatbot
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### 2. Get free API keys

- **Pinecone**: sign up free at https://app.pinecone.io → copy your API key
- **Groq**: sign up free at https://console.groq.com/keys → copy your API key

### 3. Configure environment

```bash
cp .env.example .env
# then edit .env and paste in your PINECONE_API_KEY and GROQ_API_KEY
```

### 4. Ingest the eBook (one-time)

```bash
python -m app.ingest
```

This downloads the PDF, chunks it (~50 chunks), embeds each chunk locally, and
upserts everything into your Pinecone index. Re-run any time you want to rebuild
the index (chunk IDs are content-hashed, so re-running is idempotent).

### 5. Run the chatbot

**Option A — FastAPI:**
```bash
uvicorn app.api:app --reload --port 8000
```
Then POST to `http://localhost:8000/chat`:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Agentic AI?"}'
```

**Option B — Streamlit UI:**
```bash
streamlit run app/ui.py
```

### 6. Run tests

```bash
python -m pytest tests/ -v
```

Tests cover chunking logic and the full LangGraph pipeline (retrieve → generate
→ confidence) with mocked Pinecone/Groq calls, so they run offline with no API
keys required.

## Project structure

```
agentic-rag-chatbot/
├── app/
│   ├── config.py       # all env vars / tunables in one place
│   ├── ingest.py        # PDF download → chunk → embed → upsert to Pinecone
│   ├── rag_graph.py     # LangGraph pipeline (retrieve/generate/confidence)
│   ├── api.py           # FastAPI /chat endpoint
│   └── ui.py             # optional Streamlit chat UI
├── tests/
│   └── test_pipeline.py  # offline unit tests (mocked network calls)
├── data/                  # downloaded PDF lands here (gitignored)
├── requirements.txt
├── .env.example
├── SAMPLE_QUERIES.md
└── README.md
```

## Notes

- The confidence threshold and top-k are tunable in `.env` — raise
  `MIN_CONFIDENCE_THRESHOLD` for stricter grounding, lower it for more lenient
  answers.
- Chunk size is character-based (800/120 overlap) rather than token-based for
  simplicity; swapping in a tokenizer-aware splitter (e.g. `langchain`'s
  `RecursiveCharacterTextSplitter`) is a straightforward upgrade.
