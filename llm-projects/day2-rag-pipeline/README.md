# Day 2 — RAG Pipeline → FastAPI Production Service

A complete RAG pipeline that evolved from a command-line script into a tested, containerized, deployed FastAPI service. Ingests PDFs, stores them in a vector database, and answers questions grounded in the document with page citations — accessible via a public REST API.

## What it does
- Loads PDFs with table-aware extraction (pdfplumber)
- Splits into chunks (500 chars, 50 overlap) using RecursiveCharacterTextSplitter
- Embeds with all-MiniLM-L6-v2 (local, free, no API cost)
- Stores in ChromaDB, one collection per document (persisted to disk / volume)
- Retrieves top-k chunks by cosine similarity
- Answers with Llama 3.1 via Groq, with page citations
- Exposed via FastAPI with `/ingest` and `/query` endpoints
- Tested with pytest covering success and failure paths
- Containerized with Docker, deployed live on Railway
- CI via GitHub Actions running the full test suite on every push

## Architecture
```
PDF → pdfplumber (text + tables) → pages
    → RecursiveCharacterTextSplitter → chunks
    → all-MiniLM-L6-v2 → vectors (384 dims)
    → ChromaDB (per-document collection) → persisted index

API request → /ingest → loads PDF, chunks, embeds, stores
            → /query  → embeds question, retrieves top-k,
                         calls Llama 3.1, returns answer + sources
```

## Stack
Python · FastAPI · Pydantic · LangChain · ChromaDB · HuggingFace Sentence Transformers · Groq API · Llama 3.1 8B · pdfplumber · pytest · Docker · Railway · GitHub Actions

## Key decisions
- RecursiveCharacterTextSplitter over fixed splitting — respects sentence boundaries
- chunk_overlap=50 — prevents context loss at chunk boundaries
- persist_directory — loads existing index instantly, no re-embedding
- temperature=0 — deterministic answers for factual Q&A

---

## Live deployment

**Live URL:** `https://llm-projects-production.up.railway.app`
**Interactive docs:** `https://llm-projects-production.up.railway.app/docs`

### Example request/response
```json
POST /query
{
  "collection_name": "attention_paper",
  "question": "What is multi-head attention?"
}
```
```json
{
  "answer": "Multi-Head Attention is described in section 3.2. (Page 4)",
  "sources": ["Page 1", "Page 12", "Page 4"]
}
```

### Run locally
```bash
source ../venv/bin/activate
uvicorn api:app --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive Swagger UI.

---

## FastAPI service (`api.py`)

REST API with request validation, error handling, and logging.

### Endpoints
| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check |
| `/ingest` | POST | Load a PDF, chunk it, embed it, store in ChromaDB |
| `/query` | POST | Ask a question against an ingested document |

### Request/response models (Pydantic)
```python
class IngestRequest(BaseModel):
    pdf_path: str

class QueryRequest(BaseModel):
    collection_name: str
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
```

### Error handling
- Missing file → `404`
- Wrong file type → `400`
- Querying an un-ingested collection → `404`
- Malformed request (missing required field) → `422` (automatic, via Pydantic)
- Unexpected failure → `500` with logged stack trace

### Logging
Every request and every Groq API call is logged with timestamps for observability:
```
2026-06-19 11:48:01 - INFO - Ingested 15 pages, 108 chunks for 'attention_paper'
2026-06-19 11:52:03 - INFO - Querying 'attention_paper': What is multi-head attention?
```

---

## Testing (`test_api.py`)

Six pytest tests covering both success and failure paths using FastAPI's `TestClient`.

| Test | What it checks |
|---|---|
| `test_health_check` | Health endpoint returns 200 and correct status |
| `test_ingest_missing_file_returns_404` | Ingesting a non-existent file fails gracefully |
| `test_ingest_non_pdf_returns_400` | Non-PDF files are rejected before processing |
| `test_query_missing_collection_returns_404` | Querying an un-ingested document returns a clear 404 |
| `test_query_missing_question_field_returns_422` | Pydantic rejects malformed requests automatically |
| `test_query_real_question_returns_valid_answer` | Full end-to-end: ingest → query → grounded answer with sources |

```bash
pytest test_api.py -v
```

Most RAG demos only test the happy path. These tests explicitly cover failure modes — missing files, wrong file types, un-ingested collections, malformed requests — so the API fails predictably with proper status codes instead of crashing.

---

## Bug #1: PDF tables silently dropped

Tested the pipeline against 3 papers (Attention Is All You Need, BERT, GPT-3) to find failure modes beyond the happy path.

**Test:** Asked "What is the size of the largest GPT-3 model in billions of parameters?" against the GPT-3 paper.

**Result:** The system answered "3B" — a hallucinated number, confidently cited to a page that did not contain it. The correct answer is 175B parameters, listed in the paper's Table 2.1.

**Root cause:** `PyPDFLoader` only extracts plain text from PDFs — it does not parse tables or figures. The retrieved chunk contained text *around* the table, not the table's actual data, so the LLM either said "I don't know" or pulled a nearby unrelated number and presented it as fact.

**Why this matters:** This is a common, underestimated failure mode in production RAG systems — any document with structured data (tables, charts, figures) silently loses that information during plain-text extraction, leading to confident but wrong answers rather than honest "I don't know" responses.

**Fix:** Replaced `PyPDFLoader` with a custom `pdfplumber`-based loader that extracts tables separately and converts them into readable `cell | cell | cell` text with explicit `[TABLE]` markers, merged back into each page's content before chunking.

**Result after fix:** Same question now correctly returns **175B**, citing the page containing Table 2.1.

---

## Evaluation — RAGAS

Built a 20-question evaluation set covering the paper end-to-end (authors, architecture, numbers, reasoning) and scored the RAG pipeline using RAGAS across 3 metrics.

**How it works:**
1. For each of the 20 questions: search ChromaDB for the top-3 relevant paragraphs (retrieval), send those paragraphs + the question to Llama to get an answer (generation), and save the question, the paragraphs used, the answer given, and the correct answer.
2. Hand all 20 records to RAGAS, which uses another LLM call as a "grader" to score each one on:
   - **Faithfulness** — does the answer match the retrieved paragraphs?
   - **Answer relevancy** — does the answer match the question?
   - **Context recall** — did the retrieved paragraphs actually contain the real answer?
3. Average all 20 scores per metric for the final numbers.

### Results

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | 0.78 | Are answers grounded in retrieved context, not hallucinated? |
| Answer Relevancy | 1.00 | Do answers actually address the question asked? |
| Context Recall | 0.51 | Did retrieval pull back the chunks containing the real answer? |

### Bug #2: missing spaces in extracted text

Initial run scored faithfulness 0.45 and relevancy 0.48 — much lower than expected. Investigated by inspecting raw extracted text and found `pdfplumber`'s default text extraction was merging words together without spaces (e.g. "describedinsection3.2" instead of "described in section 3.2"), caused by default character-spacing tolerance being too loose for this PDF's font.

**Fix:** Added `x_tolerance=1, y_tolerance=3` to `extract_text()` calls, which restored correct word spacing.

**Result after fix:** Faithfulness jumped from 0.45 → 0.78, answer relevancy from 0.48 → 1.00. The LLM couldn't properly parse run-together text, causing both hallucination and irrelevant answers — once it could read clean text, it grounded answers correctly and stayed on-topic.

### Known remaining limitation: context recall (0.51)

Context recall did not improve with the spacing fix, since it measures retrieval quality, not generation quality. Inspecting retrieved chunks showed retrieval often pulls back near-duplicate chunks (due to `chunk_overlap=50` combined with `k=3`), effectively wasting retrieval slots and missing some correct chunks roughly half the time.

**Next steps to improve (not yet implemented):** increase k to 5 for more retrieval attempts, increase chunk_size to reduce duplication from overlap, or add a deduplication step before passing chunks to the LLM.

---

## Docker

Containerized the FastAPI service so it runs identically on any machine without manual Python/dependency setup.

### Bug #3: torch pulling in CUDA/GPU packages

`sentence-transformers` pulls in `torch` as a dependency, which by default tried installing full CUDA/NVIDIA GPU packages inside the Linux container — multiple gigabytes of unnecessary downloads, since this app runs on CPU (using Groq's cloud API for inference, not local GPU compute).

**Fix:** Explicitly install CPU-only torch first:
```dockerfile
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
```

### Build and run
```bash
docker build -t rag-api .
docker run -p 8000:8000 --env-file ../.env rag-api
```

### Verified working in container
```json
POST /ingest {"pdf_path": "attention_paper.pdf"}
→ {"collection_name": "attention_paper", "pages_loaded": 15, "chunks_created": 118}

POST /query {"collection_name": "attention_paper", "question": "What is multi-head attention?"}
→ Correct, grounded answer with page citations
```

---

## Docker Compose + persistent volumes

Added `docker-compose.yml` to run the API alongside named volumes for ChromaDB, so vector data survives container restarts — not just the API process itself.

### Bug #4: crash loop on `docker-compose up`

Running `docker-compose up --build` caused the container to crash repeatedly: `RuntimeError: Cannot send a request, as the client has been closed`.

**Root cause:** the embedding model (`all-MiniLM-L6-v2`) downloads from HuggingFace Hub at container **startup** (when `HuggingFaceEmbeddings(...)` is instantiated). `restart: unless-stopped` in the compose file kept restarting the container faster than the download could finish, killing the in-progress download mid-request.

**Fix:** Pre-download the embedding model at **build time** instead of start time:
```dockerfile
RUN python -c "from langchain_community.embeddings import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')"
```
This bakes the model weights into the image itself, so the container never needs a network call to start successfully.

---

## Cloud deployment — Railway

Deployed the FastAPI service to Railway (free tier) so it's reachable from a public URL, not just localhost.

### Configuration
- **Root directory:** `llm-projects/day2-rag-pipeline` (Railway builds from this subfolder of the monorepo)
- **Build:** uses the existing `Dockerfile` directly — no separate Railway-specific config needed
- **Secret:** `GROQ_API_KEY` set in Railway's Variables tab (the local `.env` file is gitignored and never reaches Railway, so this has to be set manually per-environment)

### Bug #5: ChromaDB persistence path was hardcoded

The original code wrote vector data to a path relative to the working directory (`./chroma_db_{name}`). On Railway's default ephemeral filesystem, this data is wiped on every redeploy or restart — fine for local dev, not fine for a live demo.

**Fix:** Introduced a `CHROMA_BASE_DIR` environment variable (defaults to `.` for local dev, so nothing changes for existing setup):
```python
CHROMA_BASE_DIR = os.environ.get("CHROMA_BASE_DIR", ".")
persist_dir = f"{CHROMA_BASE_DIR}/chroma_db_{collection_name}"
```
Attached a Railway **volume** mounted at `/app/data`, and set `CHROMA_BASE_DIR=/app/data` as a Railway-only environment variable.

### Persistence verified, not assumed

1. Ingested a PDF via `/ingest` — confirmed it landed inside the mounted volume (checked via a temporary `/debug` endpoint that inspects the filesystem at runtime)
2. Manually triggered a full container restart from Railway's dashboard
3. Queried the same collection **without re-ingesting** — got back the correct, grounded answer, proving the vector data survived the restart

The first verification attempt actually failed, because that test's data had been written *before* the `CHROMA_BASE_DIR` env var was fully applied — it was sitting in the old ephemeral path, not the volume. Re-ingesting after confirming the env var was live, then restarting again, gave a clean pass.

---

## Continuous integration — GitHub Actions

Added a GitHub Actions workflow (`.github/workflows/pytest.yml`) that runs the full pytest suite automatically on every push to `main`.

```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
```

The workflow checks out the repo, installs Python 3.11 and dependencies, injects `GROQ_API_KEY` from GitHub's encrypted Secrets (needed because one test does a real end-to-end ingest + query against Groq), and runs `pytest -v`.

**Verified passing — all 6 tests green on a real CI run:**
```
test_api.py::test_health_check PASSED
test_api.py::test_ingest_missing_file_returns_404 PASSED
test_api.py::test_ingest_non_pdf_returns_400 PASSED
test_api.py::test_query_missing_collection_returns_404 PASSED
test_api.py::test_query_missing_question_field_returns_422 PASSED
test_api.py::test_query_real_question_returns_valid_answer PASSED
```

### Bug #6: monorepo path mismatch

The workflow's `working-directory` step kept failing with "No such file or directory," even though the same path looked correct from local git commands. Root cause: this repo's actual git root sits one level above where the `llm-projects` folder name suggests — `git ls-files` (which shows real tracked paths, unlike `git ls-tree -d` checked at the wrong level) confirmed the true path is `llm-projects/day2-rag-pipeline/api.py`. Once `working-directory` matched that real path, the workflow ran cleanly.

---

## Bonus — embedding similarity demo (`similarity.py`)

Standalone script validating that the embedding model captures semantic meaning, not just keyword overlap, using cosine similarity across 7 sentence pairs.

| Comparison | Score | Interpretation |
|---|---|---|
| cat sat vs kitten rested | High (~0.75) | Different words, same meaning |
| cat sat vs quantum physics | Low (~0.12) | Unrelated meaning |
| attention mechanism vs multi-head attention | High (~0.68) | Same domain, related concept |
| attention mechanism vs quantum physics | Low (~0.15) | Unrelated meaning |

```bash
python similarity.py
```

Confirms the retrieval layer works on meaning rather than literal word matching — the core assumption behind why RAG retrieval is more powerful than keyword search.

---

## Files in this project
```
day2-rag-pipeline/
├── rag_pipeline.py     # standalone CLI RAG pipeline
├── similarity.py       # embedding similarity demo
├── api.py              # FastAPI service (/health, /ingest, /query)
├── test_api.py         # pytest test suite
├── eval_set.py          # 20-question RAGAS evaluation set
├── run_eval.py          # RAGAS evaluation runner
├── Dockerfile
├── docker-compose.yaml
├── attention_paper.pdf
├── bert_paper.pdf
├── gpt3_paper.pdf
└── README.md
```

## What I learned
- Chunking strategy directly affects retrieval quality
- Cosine similarity finds meaning, not just keywords
- Plain-text PDF extraction silently fails on tables — table-aware parsing is necessary for documents with structured data
- FastAPI + Pydantic gives automatic request validation, eliminating manual input-checking code
- Logging every request/LLM call is essential for debugging production issues
- Testing failure paths matters as much as testing the happy path
- Evaluation isn't just a vanity metric — it surfaced a second, distinct bug (text-spacing) that manual testing had missed entirely
- Faithfulness and relevancy are generation-quality metrics; context recall is a retrieval-quality metric — they can move independently of each other
- A RAG system can score well on "did you answer correctly" while still having a real retrieval weakness hiding underneath
- Cloud filesystems are ephemeral by default — persistence has to be deliberately configured and then actually verified, not just assumed from a successful deploy log
- A repo's real git root and tracked file paths can differ from what a folder's name implies — `git ls-files` is the ground truth when paths feel inconsistent across tools

## Next
Project 1 (RAG pipeline) is fully complete: built, debugged, evaluated, tested, containerized, deployed live to Railway with persistent storage, and wired into CI via GitHub Actions. Moving to resume/LinkedIn updates and first job applications, then Project 2 — fine-tuning a model with LoRA/QLoRA.