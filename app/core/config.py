"""
app/core/config.py
------------------
Centralised configuration loaded from .env.
All tunable constants live here — never hardcode secrets in other modules.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # loads .env from project root


class Settings:
    # ── LLM ─────────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Default provider/model (overridable per-request from the UI)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

    # ── Chunking ─────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # ── Retrieval ────────────────────────────────────────────────────────────
    RETRIEVAL_CANDIDATES: int = int(os.getenv("RETRIEVAL_CANDIDATES", "50"))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))

    # ── Embedding ────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIM: int = 384  # dimension for all-MiniLM-L6-v2

    # ── Paths ────────────────────────────────────────────────────────────────
    DATA_DIR: str = os.getenv("DATA_DIR", "Data")
    PDFS_DIR: str = os.path.join(os.getenv("DATA_DIR", "Data"), "PDFS")
    CHUNKS_DIR: str = os.path.join(os.getenv("DATA_DIR", "Data"), "Processed_Chunks")
    VECTOR_STORE_DIR: str = os.path.join(os.getenv("DATA_DIR", "Data"), "VectorStore")
    CHUNKS_FILE: str = os.path.join(os.path.join(os.getenv("DATA_DIR", "Data"), "Processed_Chunks"), "chunks.jsonl")


settings = Settings()
