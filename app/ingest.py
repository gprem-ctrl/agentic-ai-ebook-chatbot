"""
Ingestion pipeline:
  1. Download the Agentic AI eBook PDF (if not already present locally)
  2. Extract text page by page and split it into overlapping chunks
  3. Generate embeddings for each chunk (sentence-transformers, local + free)
  4. Upsert (id, vector, metadata) into a Pinecone index

Run:
    python -m app.ingest             # ingest (safe to re-run with the same settings)
    python -m app.ingest --reset     # delete the Pinecone index first, then ingest
"""
import argparse
import hashlib
import os
import time
from typing import Dict, List

import pdfplumber
import requests

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
    """Extract text page by page with pdfplumber (handles custom embedded font
    encodings more reliably than pypdf, which can emit U+FFFD replacement
    characters for some fonts in this document)."""
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
    Sliding-window chunker over each page's text.

    - Chunk ends and starts snap to spaces so words are not cut.
    - The loop stops at the end of the page, so no redundant tail chunk
      (one fully contained in the previous chunk's overlap) is produced.
    - The source page number is kept in metadata.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    for page in pages:
        text = page["text"]
        n = len(text)
        start = 0
        while start < n:
            end = min(start + chunk_size, n)
            if end < n:
                space = text.rfind(" ", start + overlap + 1, end)
                if space != -1:
                    end = space
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"text": chunk, "page": page["page"]})
            if end >= n:
                break
            start = end - overlap
            nxt = text.find(" ", start, end)
            if nxt != -1:
                start = nxt + 1
    print(f"[ingest] Produced {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks


def embed_chunks(chunks: List[Dict], model_name: str = config.EMBEDDING_MODEL_NAME) -> List[Dict]:
    """Attach a dense embedding vector to each chunk."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    dim = get_dim()
    if dim != config.EMBEDDING_DIM:
        raise RuntimeError(
            f"Embedding model outputs {dim}-dim vectors but EMBEDDING_DIM={config.EMBEDDING_DIM}. "
            "Update EMBEDDING_DIM and recreate the Pinecone index (python -m app.ingest --reset)."
        )

    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector.tolist()
    print(f"[ingest] Embedded {len(chunks)} chunks with dim={dim}")
    return chunks


def _pinecone_client():
    from pinecone import Pinecone

    if not config.PINECONE_API_KEY:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Create a free index at https://app.pinecone.io "
            "and add the key to your .env file (see .env.example)."
        )
    return Pinecone(api_key=config.PINECONE_API_KEY)


def reset_index() -> None:
    """Delete the Pinecone index if it exists (used by --reset)."""
    pc = _pinecone_client()
    name = config.PINECONE_INDEX_NAME
    if name in [idx["name"] for idx in pc.list_indexes()]:
        print(f"[ingest] Deleting Pinecone index '{name}'")
        pc.delete_index(name)
        while name in [idx["name"] for idx in pc.list_indexes()]:
            time.sleep(1)


def get_pinecone_index():
    from pinecone import ServerlessSpec

    pc = _pinecone_client()
    existing = [idx["name"] for idx in pc.list_indexes()]

    if config.PINECONE_INDEX_NAME not in existing:
        print(f"[ingest] Creating Pinecone index '{config.PINECONE_INDEX_NAME}'")
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
        while not pc.describe_index(config.PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)

    return pc.Index(config.PINECONE_INDEX_NAME)


def upsert_chunks(chunks: List[Dict], batch_size: int = 100) -> None:
    index = get_pinecone_index()
    vectors = []
    for chunk in chunks:
        # Include the page so identical text on different pages doesn't collide.
        chunk_id = hashlib.md5(f"{chunk['page']}:{chunk['text']}".encode("utf-8")).hexdigest()
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


def run_ingestion(reset: bool = False) -> None:
    if reset:
        reset_index()
    pdf_path = download_pdf()
    pages = extract_text(pdf_path)
    chunks = chunk_text(pages)
    chunks = embed_chunks(chunks)
    upsert_chunks(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest the Agentic AI eBook into Pinecone.")
    parser.add_argument("--reset", action="store_true",
                        help="Delete the Pinecone index first, then re-ingest.")
    run_ingestion(reset=parser.parse_args().reset)
