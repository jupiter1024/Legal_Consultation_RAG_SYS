# LinuxGPT — Tech Spec

**System:** RAG-based Linux Documentation Assistant   
**Stack:** FastAPI 0.115.6 · FAISS-CPU 1.8.0 · SentenceTransformer 3.3.1 · LangChain 0.3.18 · LangChain-Groq 0.2.4 · Python 3.11 · Docker  
**Default LLM:** `groq / llama-3.1-8b-instant` (temperature = 0)  
**Knowledge Base PDFs:** `coreutils.pdf` · `thelinuxcommandline.pdf`

---

## Executive Summary

LinuxGPT is a Retrieval-Augmented Generation (RAG) system that answers Linux command questions grounded exclusively in uploaded PDF documentation (Linux books, man pages). It eliminates LLM hallucination of command flags by enforcing a strict context-only policy: every answer is generated solely from retrieved document chunks and carries inline citations (`[Source: filename, p.N]`).

At query time the pipeline runs three stages: **(1) semantic retrieval** — the raw user question is encoded by `all-MiniLM-L6-v2` (the same model used at ingestion), L2-normalised to a 384-dim unit vector, and searched against the FAISS `IndexFlatIP` index to return the top `RETRIEVAL_CANDIDATES = 50` chunks by cosine similarity; **(2) cross-encoder re-ranking** — each `(question, chunk)` pair is scored jointly by `ms-marco-MiniLM-L-6-v2` and the top `RERANK_TOP_K = 5` are selected; **(3) generation** — a Groq-hosted LLaMA 3.1 model produces a Markdown answer with `bash` code blocks and source citations from the retrieved context.

An **LLM Factory (Strategy Pattern)** abstracts over three providers — Groq, OpenAI, Ollama — selectable via `.env` or a per-request `provider`/`model` field in the `/chat` JSON payload, with zero code changes required.

---

## System Architecture

```
INGESTION PIPELINE
──────────────────────────────────────────────────────────────────────
 Data/PDFS/
      │   ├── coreutils.pdf            (1.2 MB — GNU coreutils manual)
      │   └── thelinuxcommandline.pdf  (7.5 MB — The Linux Command Line book)
      │
      ▼  pdfplumber==0.11.4  (two-pass extraction)
      │   Pass 1 → build line_frequency map across all pages
      │   Pass 2 → drop lines appearing on >30 % of pages (headers/footers)
      │            drop pages with <100 chars (image-only)
      ▼
 RecursiveCharacterTextSplitter  [chunk_size=1000, chunk_overlap=200]
      │   separators = ["\n\n", "\n", ". ", " ", ""]
      ▼
 SentenceTransformer('all-MiniLM-L6-v2')  →  float32[384]
      │   faiss.normalize_L2(embedding)     →  unit vector
      ▼
 faiss.IndexFlatIP  (inner product on unit vectors = cosine similarity)
      │
      ├── Data/VectorStore/linux_helper.index   (FAISS binary)
      └── Data/VectorStore/linux_helper.pkl     (metadata list)
          Each entry: { text, source, page, chunk_index, chunk_id }

      Chunks saved to: Data/Processed_Chunks/chunks.jsonl  (append mode)

QUERY PIPELINE
──────────────────────────────────────────────────────────────────────
 POST /chat  { message, provider?, model? }
      │
      ▼  Stage 1 — Semantic Embedding & Retrieval
      │   SentenceTransformer('all-MiniLM-L6-v2').encode(user_question)
      │   → float32[384]  →  faiss.normalize_L2()  →  unit vector
      │
 VectorStore.retrieve_candidates(question, limit=50)
      │   IndexFlatIP.search(query_vec, k=50)
      │   → top-50 chunks ranked by cosine similarity score
      ▼
      Stage 2 — Cross-Encoder Re-ranking
 CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
      │   .predict([(question, chunk_text)] × 50)
      │   each pair scored jointly → sort descending → top 5
      ▼
      Stage 3 — LLM Generation
 LLMFactory.create(provider, model)
      │   ┌─ "groq"   → ChatGroq(model, temperature=0, api_key=GROQ_API_KEY)
      │   ├─ "openai" → ChatOpenAI(model, temperature=0, api_key=OPENAI_API_KEY)
      │   └─ "ollama" → ChatOllama(model, base_url=OLLAMA_BASE_URL)
      ▼
 LINUX_SYSTEM_PROMPT + Retrieved Context + User Question
      ▼
 ChatResponse { answer: Markdown, sources: [SourceInfo], provider, model }
      │  SourceInfo fields: source, page, text_snippet, cosine_score, rerank_score
      ▼
 FastAPI 0.115.6  →  app/static/  (LinuxGPT terminal UI)
```

---

## API Documentation

Base URL: `http://localhost:8001`  
Interactive docs (auto-generated): `http://localhost:8001/docs`

---

### `GET /health`

Liveness probe. Confirms the server is up and reports how many vectors are loaded.

**Response `200 OK`**
```json
{
  "status": "ok",
  "vector_store_size": 1847
}
```

---

### `GET /llm/providers`

Returns the complete provider + model registry from `PROVIDER_REGISTRY` in `llm_factory.py`. Called by the frontend on page load to populate the model-selector dropdown.

**Response `200 OK`**
```json
[
  {
    "provider": "groq",
    "models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"]
  },
  {
    "provider": "openai",
    "models": ["gpt-4o-mini", "gpt-4o"]
  },
  {
    "provider": "ollama",
    "models": ["llama3", "mistral", "phi3", "codellama"]
  }
]
```

---

### `POST /upload`

Accepts a PDF, runs `process_document()` (parse + chunk → `chunks.jsonl`), then calls `vectorize_and_upload()` (embed + FAISS). Skips the file silently if it is already in the index.

**Request** — `multipart/form-data`
```
Content-Type: multipart/form-data
field name:   file
value:        <PDF binary>
```

**Response `200 OK`**
```json
{
  "message": "Successfully processed and indexed linux_commands_bible.pdf.",
  "filename": "linux_commands_bible.pdf",
  "chunks_added": 312
}
```

**Response `200 OK`** (already indexed — skip)
```json
{
  "message": "linux_commands_bible.pdf is already indexed — skipping.",
  "filename": "linux_commands_bible.pdf",
  "chunks_added": 0
}
```

**Error `400`**
```json
{ "detail": "Only PDF files are accepted." }
```

**Error `500`**
```json
{ "detail": "<exception message>" }
```

---

### `POST /chat`

Runs the full RAG pipeline. `provider` and `model` are optional — if omitted the server reads `LLM_PROVIDER` and `LLM_MODEL` from `.env`. When provided they override the defaults for that single request only (used by the frontend model-selector dropdown).

**Request `application/json`**
```json
{
  "message":  "How do I change file permissions?",
  "provider": "groq",
  "model":    "llama-3.1-8b-instant"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | `string` | ✅ | The user's Linux question |
| `provider` | `string` | ❌ | `groq` \| `openai` \| `ollama` — defaults to `.env` |
| `model` | `string` | ❌ | Model ID from the provider's list — defaults to `.env` |

**Response `200 OK`**
```json
{
  "answer": "## Changing File Permissions\n\nTo change file permissions in Linux, use the `chmod` command. Permissions are defined for the owner, the group, and others.\n\n```bash\nchmod 755 filename\n```\n\n- `7` (rwx): Owner can read, write, and execute.\n- `5` (r-x): Group and others can read and execute. [Source: thelinuxcommandline.pdf, p.94]\n\n💡 **Pro Tip**: Use `chmod +x script.sh` to quickly make a script executable without calculating octal values.",
  "sources": [
    {
      "source":       "thelinuxcommandline.pdf",
      "page":         94,
      "text_snippet": "chmod — Change file mode. The chmod command is used to change the permissions of a file or directory. It supports both octal and symbolic modes...",
      "rerank_score": 0.8942
    }
  ],
  "provider": "groq",
  "model":    "llama-3.1-8b-instant"
}
```

| Response field | Type | Description |
|---|---|---|
| `answer` | `string` | Markdown-formatted answer with `bash` code blocks and inline citations |
| `sources` | `SourceInfo[]` | Top-5 re-ranked chunks used as context |
| `sources[].source` | `string` | PDF filename |
| `sources[].page` | `int` | Page number |
| `sources[].text_snippet` | `string` | First 200 chars of the chunk |
| `sources[].rerank_score` | `float` | Cross-encoder relevance score |
| `provider` | `string` | Provider actually used |
| `model` | `string` | Model actually used |

**Error `500`**
```json
{ "detail": "<exception message>" }
```

---

## Embedding Model & Chunking Strategy Justification

### Embedding Model — `sentence-transformers/all-MiniLM-L6-v2`

| Property | Value | Rationale |
|---|---|---|
| Output dimension | **384** | Compact — fast FAISS search, low RAM (1 M vectors ≈ 1.5 GB) |
| Architecture | MiniLM-L6 (distilled from BERT-large) | Runs on CPU — no GPU needed in Docker |
| Training corpora | MS MARCO, NLI, STS benchmarks | Strong semantic alignment for English technical Q&A |
| Licence | Apache 2.0 | No API key, zero cost, no rate limits |
| Version pinned | `sentence-transformers==3.3.1` | Reproducible builds |

**Cosine similarity** is computed via `faiss.IndexFlatIP` on L2-normalised vectors:

```
cos_sim(a, b) = dot(a/|a|, b/|b|) = inner_product(norm(a), norm(b))
```

This is exact (no approximation), making it suitable for a corpus of 1,000–100,000 chunks without an ANN index.

### Chunking Strategy — `RecursiveCharacterTextSplitter`

```python
chunk_size    = 1000   # characters  ≈ 150–200 words
chunk_overlap = 200    # characters  ≈ 1–2 sentences
separators    = ["\n\n", "\n", ". ", " ", ""]
```

| Parameter | Value | Why |
|---|---|---|
| `chunk_size = 1000` | ≈ one command + all its flag descriptions | Large enough for full context, small enough for focused embeddings |
| `chunk_overlap = 200` | ≈ 1–2 bridge sentences | Prevents meaning loss when a key sentence straddles a boundary |
| `separators` (recursive) | Paragraph → line → sentence → word | Splits at the most natural boundary available, never mid-word |

**Pre-processing before chunking** (two-pass PDF extraction):
- Lines appearing on more than **30 %** of pages are identified as headers/footers and removed
- Pages with fewer than **100 characters** of extracted text are skipped (image-only pages)
- Consecutive whitespace is collapsed before chunking

This keeps noise vectors out of the FAISS index and prevents the retriever from returning irrelevant boilerplate text as top candidates.

---

## Docker Deployment Instructions

### Prerequisites
- Docker Desktop ≥ 4.x installed and running
- Groq API key — free at [console.groq.com](https://console.groq.com)

### Step 1 — Configure `.env`

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```bash
# .env
GROQ_API_KEY=gsk_your_key_here

# Active LLM (change anytime — zero code edits needed)
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant

# Retrieval parameters
RETRIEVAL_CANDIDATES=50
RERANK_TOP_K=5

# Chunking parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Embedding
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Step 2 — PDFs Are Already in Place

The project ships with two Linux documentation PDFs in `Data/PDFS/`:

```
Data/PDFS/
├── coreutils.pdf            # GNU Coreutils manual (1.2 MB)
└── thelinuxcommandline.pdf  # The Linux Command Line book (7.5 MB)
```

To add more books, simply copy them into `Data/PDFS/`:

```bash
cp ~/another_linux_book.pdf Data/PDFS/
```

### Step 3 — Build & Start the Container

```bash
docker-compose up --build
```

The container exposes port `8001`. First startup downloads `all-MiniLM-L6-v2` and `ms-marco-MiniLM-L-6-v2` from HuggingFace Hub (~200 MB total, cached in the image layer after first build).

### Step 4 — Ingest PDFs into FAISS

Open a second terminal while the container is running:

```bash
docker-compose exec api python -m scripts.ingest
```

Expected output:
```
LinuxGPT — Data Ingestion Pipeline
Found 2 PDF(s):
  - coreutils.pdf
  - thelinuxcommandline.pdf

[Parser] Processing: coreutils.pdf
  Extracted 108 pages with content.
  Generated 623 chunks.
  Saved to Data/Processed_Chunks/chunks.jsonl

[Parser] Processing: thelinuxcommandline.pdf
  Extracted 525 pages with content.
  Generated 2891 chunks.
  Saved to Data/Processed_Chunks/chunks.jsonl

[VectorStore] Added 3514 vectors. Total: 3514.
Done. Index size: 3514 total.
```

### Step 5 — Open the App

Navigate to **http://localhost:8001**

- **LLM Engine** dropdown (sidebar) — switch provider and model per query
- **Knowledge Base** panel — upload additional PDFs without restart
- **Quick Queries** buttons — pre-filled common Linux questions

### Verify Deployment

```bash
# Health check
curl http://localhost:8001/health
# {"status":"ok","vector_store_size":3514}

# List providers (matches PROVIDER_REGISTRY in llm_factory.py)
curl http://localhost:8001/llm/providers

# Test a chat query against the real books
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the difference between a hard link and a symbolic link?","provider":"groq","model":"llama-3.1-8b-instant"}'
```

### Force Full Re-index

```bash
# Wipes linux_helper.index + linux_helper.pkl, re-processes all PDFs
docker-compose exec api python -m scripts.ingest --force
```

> **Persistence note:** `./Data` is bind-mounted as a Docker volume (`./Data:/app/Data` in `docker-compose.yml`). The FAISS index survives `docker-compose down` and container rebuilds — ingestion only needs to run once per PDF.

---

## Retrieval Accuracy & Hallucination Analysis

RAG systems are prone to hallucination and poor retrieval. Below is an evaluation of LinuxGPT's retrieval performance based on 3 edge-case queries where the system's architecture reached its limits.

### Edge Case 1: Negative Constraints
**Query:** *"Show me how to list files WITHOUT using the ls command."*

*   **Failure Mode:** Poor Retrieval (Semantic Noise).
*   **Result:** The system retrieved chunks heavily describing the `ls` command and its flags. The LLM then attempted to explain `ls` despite the "WITHOUT" constraint.
*   **Architectural Reason:** The embedding model (`all-MiniLM-L6-v2`) is a bi-encoder that maps "list files" to a vector space dominated by `ls`. It lacks the logical reasoning to penalize terms like "WITHOUT" or "NOT" effectively. The re-ranker also prioritized "list files" similarity over the negative instruction.

### Edge Case 2: Broad Synthesis (Global Context)
**Query:** *"Summarize all security-related commands mentioned in the documentation."*

*   **Failure Mode:** Incomplete Information Retrieval.
*   **Result:** The system retrieved the Table of Contents pages and a single chapter introduction on file permissions (chmod/chown), missing security topics like `ssh`, `iptables`, or `sudo` mentioned elsewhere.
*   **Architectural Reason:** The system uses a fixed `RERANK_TOP_K = 5`. Five chunks (approx. 5000 characters) are insufficient to capture information scattered across 600+ pages of documentation. Without a hierarchical retrieval strategy or an agentic "map-reduce" approach, broad summaries will always be lossy.

### Edge Case 3: Complex Multi-Flag Coordination
**Query:** *"How do I find empty files modified exactly 3 days ago only in the current directory?"*

*   **Failure Mode:** Partial Context Retrieval.
*   **Result:** The system retrieved chunks for the `find` command, but the specific flags `-empty`, `-mtime 3`, and `-maxdepth 1` were spread across different sections of the manual. The retriever found a chunk for `-mtime` but missed the one explaining `-maxdepth`.
*   **Architectural Reason:** The `chunk_size` of 1000 characters often splits a long manual page (like `find`) into 10+ chunks. Since retrieval is independent, the system might not "see" all required flags if they aren't co-located in the top 5 chunks. A larger chunk size or "context stitching" would be required to solve this.
