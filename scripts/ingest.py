"""
scripts/ingest.py
-----------------
One-shot CLI script to parse all PDFs in data/pdfs/ and build the FAISS index.

Usage:
  python -m scripts.ingest           # skip already-indexed files
  python -m scripts.ingest --force   # wipe index and re-process everything
"""

import os
import sys

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.services.parser import process_document
from app.services.vectorstore import VectorStore


def ingest_all(force: bool = False) -> None:
    print("=" * 60)
    print("  LinuxGPT — Data Ingestion Pipeline")
    print("=" * 60)

    vs = VectorStore()

    if force:
        print("\n[Ingest] --force: clearing existing index and chunks...")
        vs.clear()
        if os.path.exists(settings.CHUNKS_DIR):
            import shutil
            shutil.rmtree(settings.CHUNKS_DIR)
        os.makedirs(settings.CHUNKS_DIR, exist_ok=True)

    if not os.path.exists(settings.PDFS_DIR):
        print(f"\n[Ingest] PDF directory not found: {settings.PDFS_DIR}")
        print("  Create it and drop your Linux books/man-page PDFs in there, then re-run.")
        return

    pdf_files = [f for f in os.listdir(settings.PDFS_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"\n[Ingest] No PDF files found in {settings.PDFS_DIR}")
        return

    print(f"\n[Ingest] Found {len(pdf_files)} PDF(s):")
    for f in pdf_files:
        print(f"  - {f}")

    # ── Parse each PDF ────────────────────────────────────────────────────────
    total_chunks = 0
    for filename in pdf_files:
        if not force and vs.document_exists(filename):
            print(f"\n[Ingest] {filename} is already indexed — skipping.")
            continue

        pdf_path = os.path.join(settings.PDFS_DIR, filename)
        chunks = process_document(pdf_path, settings.CHUNKS_DIR)
        total_chunks += chunks

    # ── Vectorise ─────────────────────────────────────────────────────────────
    if os.path.exists(settings.CHUNKS_FILE):
        print(f"\n[Ingest] Embedding and indexing {total_chunks} new chunk(s)...")
        added = vs.vectorize_and_upload(settings.CHUNKS_FILE, skip_existing=not force)
        print(f"\n[Ingest] Done. {added} vectors added. Index size: {vs.size} total.")
    else:
        print("\n[Ingest] No chunks file produced — nothing to vectorise.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    force = "--force" in sys.argv
    ingest_all(force=force)
