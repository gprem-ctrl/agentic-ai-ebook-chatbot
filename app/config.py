"""
Centralized configuration for the Agentic AI RAG Chatbot.
All values are read from environment variables (see .env.example).
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Source document ---
PDF_URL = os.getenv("PDF_URL", "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf")
PDF_LOCAL_PATH = os.getenv("PDF_LOCAL_PATH", "data/Ebook-Agentic-AI.pdf")

# --- Embeddings ---
# Local, free, no API key required (runs on CPU).
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))  # matches MiniLM-L6-v2

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(
        f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be smaller than CHUNK_SIZE ({CHUNK_SIZE})."
    )

# --- Vector DB: Pinecone ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-ebook")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# --- LLM (generation) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")

# --- Retrieval / grounding ---
TOP_K = int(os.getenv("TOP_K", "4"))
# Below this similarity score, we treat the question as out-of-scope
# rather than letting the LLM guess.
MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.35"))

OUT_OF_SCOPE_MESSAGE = (
    "I couldn't find grounded information for that in the Agentic AI eBook, "
    "so I won't guess. Please ask something covered by the document."
)
