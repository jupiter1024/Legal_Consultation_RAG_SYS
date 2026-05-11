"""
app/services/rag.py
-------------------
Two-stage RAG pipeline:
  1. Embed the raw user query with SentenceTransformer → FAISS cosine similarity → top-N candidates
  2. Cross-Encoder re-rank candidates → top-K
  3. LLM generates a Markdown answer from the retrieved context

Retrieval uses the user's question directly as the embedding input — the same
all-MiniLM-L6-v2 model that encoded the chunks at ingestion time encodes the query,
ensuring the vector space is consistent and the cosine similarity is meaningful.
"""

from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.services.llm_factory import LLMFactory
from app.services.vectorstore import VectorStore


# ── System Prompt ─────────────────────────────────────────────────────────────

LINUX_SYSTEM_PROMPT = """\
You are **LinuxGPT**, a world-class Linux systems expert and documentation assistant.
You have access to retrieved excerpts from official Linux command-line books and man pages.

## Responsibilities
1. **Answer from context only.** Never use knowledge outside the provided excerpts.
2. **Commands in code blocks.** Wrap every command and file path in a ```bash block```.
3. **Explain flags.** When a flag is mentioned in the context, include a brief explanation.
4. **Numbered steps.** If the task requires multiple actions, use a numbered list.
5. **Cite your sources.** Append `[Source: <filename>, p.<page>]` after each relevant statement.
6. **Honest fallback.** If the context does not contain the answer, respond with:
   > "I don't have documentation on that topic in the current knowledge base."
7. **No hallucination.** Never invent command flags, behaviours, or file paths.

## Output Format
- Use Markdown: headers, code blocks, bullet lists.
- Keep answers concise but complete.
- Close with a "💡 **Pro Tip**" if a related best practice appears in the context.

---
**Retrieved Context:**
{context}
"""


# ── RAG Engine ────────────────────────────────────────────────────────────────

class LinuxRAG:
    """
    Stateful RAG engine.

    Startup (once):
      - VectorStore loads FAISS index + SentenceTransformer embedding model
      - CrossEncoder re-ranker is loaded into memory

    Per-request:
      - User query is embedded with SentenceTransformer (same model used at ingestion)
      - L2-normalised query vector is searched against the FAISS IndexFlatIP
        (inner product on unit vectors = cosine similarity)
      - Top-N candidates are re-ranked by CrossEncoder
      - Top-K chunks are passed as context to the LLM (created fresh via LLMFactory
        so provider/model can change per request from the UI dropdown)
    """

    def __init__(self):
        self.vector_store = VectorStore()

        print("[RAG] Loading Cross-Encoder re-ranker (ms-marco-MiniLM-L-6-v2)...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", LINUX_SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        print("[RAG] Ready.")

    def get_response(
        self,
        question: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline.

        Step 1 — Embed & Retrieve:
            The raw user question is encoded by the same SentenceTransformer
            (all-MiniLM-L6-v2) used to embed the document chunks at ingestion.
            The resulting 384-dim float32 vector is L2-normalised and searched
            against the FAISS IndexFlatIP to return the top RETRIEVAL_CANDIDATES
            chunks by cosine similarity score.

        Step 2 — Re-rank:
            The cross-encoder (ms-marco-MiniLM-L-6-v2) scores every
            (question, chunk_text) pair jointly, capturing deeper semantic
            relevance than cosine similarity alone.  Top RERANK_TOP_K survive.

        Step 3 — Generate:
            The top chunks are assembled as context and passed with the original
            question to the LLM (created via LLMFactory for per-request switching).
        """
        _provider = provider or settings.LLM_PROVIDER
        _model    = model    or settings.LLM_MODEL

        llm = LLMFactory.create(_provider, _model)

        # ── Step 1: Embed user query → FAISS cosine similarity search ─────────
        print(f"[RAG] Embedding query: {question!r}")
        candidates = self.vector_store.retrieve_candidates(
            question,                          # raw user question — same model as chunks
            limit=settings.RETRIEVAL_CANDIDATES
        )
        print(f"[RAG] Retrieved {len(candidates)} candidates "
              f"(top cosine score: {candidates[0]['score']:.4f})" if candidates else
              "[RAG] No candidates found.")

        if not candidates:
            return {
                "answer": (
                    "I don't have documentation on that topic in the current knowledge base. "
                    "Please upload relevant Linux documentation PDFs and try again."
                ),
                "sources": [],
                "provider": _provider,
                "model": _model,
            }

        # ── Step 2: Cross-Encoder re-ranking ─────────────────────────────────
        top_results = self._rerank(candidates, question, top_k=settings.RERANK_TOP_K)

        # ── Step 3: Context assembly ──────────────────────────────────────────
        context_parts: List[str] = []
        sources: List[Dict[str, Any]] = []

        for res in top_results:
            context_parts.append(res["text"])
            meta = res["metadata"]
            sources.append({
                "source":        meta.get("source", "Unknown"),
                "page":          meta.get("page", 0),
                "text_snippet":  res["text"][:200] + "...",
                "rerank_score":  round(res.get("rerank_score", 0.0), 4),
                "cosine_score":  round(res.get("score", 0.0), 4),
            })

        context_text = "\n\n---\n\n".join(context_parts)

        # ── Step 4: LLM generation ────────────────────────────────────────────
        response = (self.answer_prompt | llm).invoke(
            {"context": context_text, "question": question}
        )

        return {
            "answer":   response.content,
            "sources":  sources,
            "provider": _provider,
            "model":    _model,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _rerank(
        self,
        candidates: List[Dict[str, Any]],
        question: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Score each (question, chunk_text) pair with the cross-encoder and
        return the top_k entries sorted by rerank_score descending.

        The cross-encoder sees the full pair jointly (unlike bi-encoder cosine
        similarity which encodes query and chunk independently), making it a
        much more accurate relevance signal.
        """
        if not candidates:
            return []

        print(f"[RAG] Re-ranking {len(candidates)} candidates...")
        pairs  = [[question, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)

        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        print(f"[RAG] Top-{top_k} re-rank scores: "
              f"{[round(r['rerank_score'], 3) for r in ranked[:top_k]]}")
        return ranked[:top_k]
