"""
app/models/schemas.py
---------------------
Pydantic models (data contracts) for all API request/response payloads.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Request Models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's Linux question.")
    provider: Optional[str] = Field(
        None,
        description="LLM provider to use for this request. Falls back to .env default."
    )
    model: Optional[str] = Field(
        None,
        description="LLM model to use for this request. Falls back to .env default."
    )


# ── Response Models ──────────────────────────────────────────────────────────

class SourceInfo(BaseModel):
    source: str = Field(..., description="Filename of the source document.")
    page: int = Field(..., description="Page number within the source document.")
    text_snippet: str = Field(..., description="First 200 characters of the retrieved chunk.")
    cosine_score: float = Field(..., description="FAISS cosine similarity score (stage 1).")
    rerank_score: float = Field(..., description="Cross-encoder re-ranking confidence score (stage 2).")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="LLM-generated answer in Markdown.")
    sources: List[SourceInfo] = Field(default_factory=list, description="Retrieved sources used.")
    provider: str = Field(..., description="LLM provider used for this response.")
    model: str = Field(..., description="LLM model used for this response.")


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_added: int


class HealthResponse(BaseModel):
    status: str = "ok"
    vector_store_size: int


# ── LLM Provider Listing ─────────────────────────────────────────────────────

class LLMProviderModels(BaseModel):
    provider: str
    models: List[str]
