"""
app/main.py
-----------
FastAPI controller — all HTTP routes.
"""

import os
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    LLMProviderModels,
    SourceInfo,
    UploadResponse,
)
from app.services.llm_factory import LLMFactory
from app.services.parser import process_document
from app.services.rag import LinuxRAG


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.PDFS_DIR, exist_ok=True)
    os.makedirs(settings.CHUNKS_DIR, exist_ok=True)
    os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)
    app.state.rag = LinuxRAG()
    print("[Main] Application started.")
    yield
    print("[Main] Application shutting down.")


app = FastAPI(
    title="LinuxGPT — RAG API",
    description="Retrieval-Augmented Generation for Linux documentation with LLM Factory.",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check — returns vector store size."""
    return HealthResponse(
        status="ok",
        vector_store_size=app.state.rag.vector_store.size,
    )


@app.get("/llm/providers", response_model=list[LLMProviderModels], tags=["LLM"])
async def list_llm_providers():
    """Return all available LLM providers and models. Used by the UI dropdown."""
    return LLMFactory.get_providers()


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, parse it into chunks, and add to the FAISS index."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    dest_path = os.path.join(settings.PDFS_DIR, file.filename)
    with open(dest_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        if app.state.rag.vector_store.document_exists(file.filename):
            return UploadResponse(
                message=f"{file.filename} is already indexed — skipping.",
                filename=file.filename,
                chunks_added=0,
            )

        process_document(dest_path, settings.CHUNKS_DIR)
        vectors_added = app.state.rag.vector_store.vectorize_and_upload(
            settings.CHUNKS_FILE, skip_existing=True
        )

        return UploadResponse(
            message=f"Successfully processed and indexed {file.filename}.",
            filename=file.filename,
            chunks_added=vectors_added,
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """Run the RAG pipeline. provider/model are optional per-request overrides."""
    try:
        result = app.state.rag.get_response(
            question=request.message,
            provider=request.provider,
            model=request.model,
        )

        sources = [
            SourceInfo(
                source=s["source"],
                page=s["page"],
                text_snippet=s["text_snippet"],
                cosine_score=s.get("cosine_score", 0.0),
                rerank_score=s["rerank_score"],
            )
            for s in result["sources"]
        ]

        return ChatResponse(
            answer=result["answer"],
            sources=sources,
            provider=result["provider"],
            model=result["model"],
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Static Frontend (mount last so API routes take priority) ──────────────────

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
