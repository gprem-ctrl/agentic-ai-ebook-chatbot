# Agentic AI eBook RAG Chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions **strictly** from the Agentic AI eBook (Konverge AI). Built with **LangGraph**, **Pinecone**, and local sentence embeddings.

Built for the Appening Infotech AI Engineer Intern interview task.

## Features

- Downloads the source PDF, chunks it, embeds it, and stores it in Pinecone
- LangGraph pipeline with a conditional branch: `retrieve → (generate | refuse) → grade_confidence`
- Answers are grounded only in retrieved chunks. If nothing relevant is found, the bot says so instead of guessing
- Every response returns the **final answer**, the **retrieved context chunks**, and a **confidence score**
- Available as a **FastAPI** JSON API and an optional **Streamlit** chat UI

## Architecture

```
Ebook-Agentic-AI.pdf
        │  download + extract (pdfplumber)
        ▼
Chunker (800 chars, 120 overlap, page-aware)
        │  embed (all-MiniLM-L6-v2, local)
        ▼
Pinecone index (cosine, 384-dim)
        │
        ▼
┌───────────────────── LangGraph pipeline ─────────────────────┐
│                                                               │
│  retrieve ──(top score ≥ threshold)──▶ generate ──┐           │
│      │                                             ▼           │
│      └──────(top score < threshold)──▶ refuse ──▶ grade_confidence
│                                                               │
└───────────────────────────────────────────────────────────────┘
        │
        ├──▶ FastAPI  POST /chat
        └──▶ Streamlit UI
```

**How it works**

1. **Ingest:** the PDF text is split into overlapping chunks that keep their page number. Each chunk is embedded locally and upserted to Pinecone. Chunk IDs are content-hashed, so re-running ingestion is safe.
2. **Retrieve:** the question is embedded and the top-k most similar chunks are fetched from Pinecone.
3. **Route:** if the best similarity score is below `MIN_CONFIDENCE_THRESHOLD` (default `0.35`), the graph goes to `refuse` and the LLM is never called.
4. **Generate:** otherwise the chunks go to Groq (LLaMA 3) with a strict system prompt that only allows answers from the given context. If the model reports the context doesn't contain the answer, the bot refuses.
5. **Grade confidence:** the confidence score is the top Pinecone cosine similarity, or `0.0` when the bot refuses.

**Design choices**

- **Local embeddings** (`sentence-transformers`): no paid API needed for embedding, and results are reproducible.
- **Pinecone** as the vector DB (the free serverless tier is enough for a document this size).
- **Groq + LLaMA 3** for fast, free-tier generation.
- **Threshold before generation:** clearly out-of-scope questions never reach the LLM, which reduces hallucination and saves a call.

## Setup

Requires Python 3.10+.

### 1. Clone and install

```bash
git clone https://github.com/gprem-ctrl/agentic-ai-ebook-chatbot.git
cd agentic-ai-ebook-chatbot
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`sentence-transformers` installs PyTorch, so the first install is large and can take a few minutes. The first run also downloads the embedding model (about 90 MB).

### 2. Get free API keys

- **Pinecone:** https://app.pinecone.io (the default region `aws / us-east-1` is the one the free plan uses)
- **Groq:** https://console.groq.com/keys

### 3. Configure environment

```bash
cp .env.example .env
```

Open `.env` and paste your `PINECONE_API_KEY` and `GROQ_API_KEY` (both are blank in `.env.example`). All other settings have working defaults.

### 4. Ingest the eBook (one-time)

```bash
python -m app.ingest
```

This downloads the PDF, chunks it, embeds each chunk locally, and upserts everything into Pinecone. Re-running with the same settings is safe. If you change the chunk size or the embedding model, rebuild the index from scratch:

```bash
python -m app.ingest --reset
```

### 5. Run the chatbot

**Option A: FastAPI**

```bash
uvicorn app.api:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Agentic AI?"}'
```

**Option B: Streamlit UI**

```bash
streamlit run app/ui.py
```

### 6. Run tests

```bash
python -m pytest tests/ -v
```

Tests cover chunking, routing, refusal paths and the full LangGraph pipeline with mocked Pinecone/Groq calls, so they run offline with no API keys.

## API

`POST /chat`

Request:

```json
{"question": "What is Agentic AI?"}
```

Response:

```json
{
  "answer": "...",
  "retrieved_context": [
    {"text": "...", "page": 3, "score": 0.71}
  ],
  "confidence": 0.71
}
```

Out-of-scope questions (for example, "What is the capital of France?") score below the threshold. The LLM is skipped, the bot replies that the eBook doesn't cover the question, and `confidence` is `0.0`.

`GET /health` returns `{"status": "ok"}`.

## Sample queries

Six sample questions (five in-scope, one out-of-scope) are in [SAMPLE_QUERIES.md](SAMPLE_QUERIES.md). To regenerate that file with real answers, scores and top chunks from your own running server:

```bash
python scripts/generate_samples.py
```

## Project structure

```
agentic-ai-ebook-chatbot/
├── app/
│   ├── config.py       # env vars and tunables
│   ├── ingest.py       # PDF download → chunk → embed → upsert to Pinecone
│   ├── rag_graph.py    # LangGraph pipeline (retrieve / generate / refuse / confidence)
│   ├── api.py          # FastAPI /chat endpoint
│   └── ui.py           # optional Streamlit chat UI
├── scripts/
│   └── generate_samples.py
├── tests/
│   └── test_pipeline.py
├── data/               # downloaded PDF (gitignored)
├── requirements.txt
├── .env.example
├── SAMPLE_QUERIES.md
└── README.md
```

## Notes and limitations

- `MIN_CONFIDENCE_THRESHOLD` and `TOP_K` are tunable in `.env`. Raise the threshold for stricter grounding, lower it for more lenient answers.
- The confidence score reflects retrieval similarity, not factual correctness.
- Chunking is character-based (800 / 120) for simplicity. A token-aware splitter such as `RecursiveCharacterTextSplitter` would be a natural upgrade.
- Each question is answered independently; there is no conversation memory.
