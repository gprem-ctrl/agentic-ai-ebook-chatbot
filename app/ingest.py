"""
Ingestion pipeline:
  1. Download the Agentic AI eBook PDF (if not already present locally)
  2. Extract text and split it into overlapping chunks
  3. Generate embeddings for each chunk (sentence-transformers, local + free)
  4. Upsert (id, vector, metadata) into a Pinecone index

Run directly:
    python -m app.ingest
"""
import os
import time
import hashlib
from typing import List, Dict

import requests
import pdfplumber
from sentence_transformers import SentenceTransformer

from app import config


def download_pdf(url: str = config.PDF_URL, local_path: str = config.PDF_LOCAL_PATH) -> str:
    """Download the source PDF if it isn't already cached locally."""
    if os.path.exists(local_path):
        print(f"[ingest] Using cached PDF at {local_path}")
        return local_path

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    print(f"[ingest] Downloading PDF from {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(resp.content)
    print(f"[ingest] Saved PDF to {local_path} ({len(resp.content)} bytes)")
    return local_path


def extract_text(pdf_path: str) -> List[Dict]:
    """Extract text from the PDF, page by page, using pdfplumber (handles
    custom embedded font encodings -- e.g. curly quotes/bullets -- more
    reliably than pypdf, which can emit U+FFFD replacement characters for
    some fonts in this document)."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"page": i + 1, "text": text})
    print(f"[ingest] Extracted text from {len(pages)} non-empty pages")
    return pages


def chunk_text(pages: List[Dict], chunk_size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> List[Dict]:
    """
    Simple sliding-window chunker over each page's text, tracking source
    page number in metadata for traceability.
    """
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append({
                    "text": chunk.strip(),
                    "page": page["page"],
                })
            start += (chunk_size - overlap)
    print(f"[ingest] Produced {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks


def embed_chunks(chunks: List[Dict], model_name: str = config.EMBEDDING_MODEL_NAME) -> List[Dict]:
    """Attach a dense embedding vector to each chunk."""
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector.tolist()
    print(f"[ingest] Embedded {len(chunks)} chunks with dim={len(vectors[0])}")
    return chunks


def get_pinecone_index():
    from pinecone import Pinecone, ServerlessSpec

    if not config.PINECONE_API_KEY:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Create a free index at https://app.pinecone.io "
            "and add the key to your .env file (see .env.example)."
        )

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    existing = [idx["name"] for idx in pc.list_indexes()]

    if config.PINECONE_INDEX_NAME not in existing:
        print(f"[ingest] Creating Pinecone index '{config.PINECONE_INDEX_NAME}'")
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
        # Wait for the index to be ready
        while not pc.describe_index(config.PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)

    return pc.Index(config.PINECONE_INDEX_NAME)


def upsert_chunks(chunks: List[Dict], batch_size: int = 100) -> None:
    index = get_pinecone_index()
    vectors = []
    for chunk in chunks:
        chunk_id = hashlib.md5(chunk["text"].encode("utf-8")).hexdigest()
        vectors.append({
            "id": chunk_id,
            "values": chunk["embedding"],
            "metadata": {"text": chunk["text"], "page": chunk["page"]},
        })

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"[ingest] Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")

    print(f"[ingest] Done. {len(vectors)} vectors stored in Pinecone index "
          f"'{config.PINECONE_INDEX_NAME}'.")


def run_ingestion() -> None:
    pdf_path = download_pdf()
    pages = extract_text(pdf_path)
    chunks = chunk_text(pages)
    chunks = embed_chunks(chunks)
    upsert_chunks(chunks)


if __name__ == "__main__":
    run_ingestion()
