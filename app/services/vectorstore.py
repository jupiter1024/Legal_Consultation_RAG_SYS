"""
app/services/vectorstore.py
---------------------------
FAISS vector store with SentenceTransformer embeddings.
Similarity: cosine via L2-normalised vectors + IndexFlatIP (inner product).
"""

import json
import os
import pickle
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from app.core.config import settings


class VectorStore:
    def __init__(self, index_name: str = "linux_helper"):
        print("[VectorStore] Initialising FAISS index...")
        os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)

        self.index_path = os.path.join(settings.VECTOR_STORE_DIR, f"{index_name}.index")
        self.meta_path  = os.path.join(settings.VECTOR_STORE_DIR, f"{index_name}.pkl")

        print(f"[VectorStore] Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.dimension = settings.EMBEDDING_DIM

        self.index: faiss.IndexFlatIP
        self.metadata: List[Dict[str, Any]] = []

        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"[VectorStore] Saved {len(self.metadata)} vectors.")

    def _load(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"[VectorStore] Loaded existing index — {len(self.metadata)} vectors.")
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            print("[VectorStore] Initialised new FAISS IndexFlatIP (cosine similarity).")

    def clear(self) -> None:
        print("[VectorStore] Clearing index...")
        self.index    = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        for path in (self.index_path, self.meta_path):
            if os.path.exists(path):
                os.remove(path)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def document_exists(self, filename: str) -> bool:
        return any(m['metadata']['source'] == filename for m in self.metadata)

    def vectorize_and_upload(
        self,
        input_file: str,
        skip_existing: bool = False,
        clear_first: bool = False,
    ) -> int:
        """Embed JSONL chunks and add to FAISS. Returns number of vectors added."""
        if not os.path.exists(input_file):
            print(f"[VectorStore] Chunks file not found: {input_file}")
            return 0

        if clear_first:
            self.clear()

        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            print("[VectorStore] Chunks file is empty.")
            return 0

        all_embeddings: List[np.ndarray] = []
        all_meta: List[Dict[str, Any]]   = []

        for line in tqdm(lines, desc="[VectorStore] Embedding"):
            chunk_data = json.loads(line)
            source = chunk_data['metadata']['source']

            if skip_existing and self.document_exists(source):
                continue

            text = chunk_data['text']
            embedding = self.model.encode(text, show_progress_bar=False).astype('float32')
            faiss.normalize_L2(embedding.reshape(1, -1))  # L2-norm → cosine via inner product

            all_embeddings.append(embedding)
            all_meta.append({'text': text, 'metadata': chunk_data['metadata']})

        if all_embeddings:
            embeddings_np = np.array(all_embeddings).astype('float32')
            self.index.add(embeddings_np)
            self.metadata.extend(all_meta)
            self.save()
            print(f"[VectorStore] Added {len(all_embeddings)} vectors. Total: {len(self.metadata)}.")
            return len(all_embeddings)

        print("[VectorStore] No new vectors to add.")
        return 0

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Cosine similarity search via FAISS inner product on L2-normalised vectors."""
        if self.index is None or len(self.metadata) == 0:
            return []

        query_vec = self.model.encode(query, show_progress_bar=False).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, limit)

        results: List[Dict[str, Any]] = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            entry = self.metadata[idx].copy()
            entry['score'] = float(scores[0][i])
            results.append(entry)

        return results

    def retrieve_candidates(self, query: str, limit: int = settings.RETRIEVAL_CANDIDATES) -> List[Dict[str, Any]]:
        """First-stage retrieval for the two-stage re-ranking pipeline."""
        return self.search(query, limit=limit)

    @property
    def size(self) -> int:
        return len(self.metadata)
