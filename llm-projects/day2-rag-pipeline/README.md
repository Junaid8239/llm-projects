# Day 2 — RAG Pipeline (Retrieval Augmented Generation)

A complete RAG pipeline that ingests a PDF, stores it in a vector database, and answers questions grounded in the document with page citations. Zero hallucination on test queries.

## What it does
- Loads the "Attention Is All You Need" paper (15 pages)
- Splits into 93 chunks (500 chars, 50 overlap)
- Embeds with all-MiniLM-L6-v2 (local, free)
- Stores in ChromaDB (persisted to disk)
- Retrieves top-3 chunks by cosine similarity
- Answers with Llama 3.1 via Groq with page citations

## Test results
| Question | Answer | Hallucination? |
|---|---|---|
| What optimizer was used? | Adam β1=0.9, β2=0.98 (Page 6) | ✅ None |
| Who are the authors? | All 6 authors correct | ✅ None |
| What is multi-head attention? | Grounded from paper | ✅ None |

## ArchitecturePDF → PyPDFLoader → 15 pages
→ RecursiveCharacterTextSplitter → 93 chunks
→ all-MiniLM-L6-v2 → 93 vectors (384 dims)
→ ChromaDB → persisted index
Query → embed → cosine similarity → top-3 chunks
→ Llama 3.1 → grounded answer + page citation

## Stack
Python · LangChain · ChromaDB · HuggingFace Sentence Transformers · Groq API · Llama 3.1 8B

## Run it
```bash
source ../venv/bin/activate
python rag_pipeline.py
```
## Key decisions
- RecursiveCharacterTextSplitter over fixed splitting — respects sentence boundaries
- chunk_overlap=50 — prevents context loss at chunk boundaries  
- persist_directory — loads existing index instantly, no re-embedding
- temperature=0 — deterministic answers for factual Q&A

## Interview answer
"I built a RAG pipeline over the Attention Is All You Need paper. 15 pages chunked into 93 pieces with 50-char overlap, embedded with all-MiniLM-L6-v2 locally, stored in ChromaDB with persistence. At query time top-3 chunks retrieved by cosine similarity, injected into a prompt with Llama 3.1 via Groq. Tested with hallucination checks — optimizer question returned exact Adam parameters from Page 6 with citation."


## Known limitation — discovered through multi-PDF stress testing

Tested the pipeline against 3 different research papers (Attention Is All You Need, BERT, GPT-3) to find failure modes beyond the happy path.

### Failure case: PDF tables and figures are silently dropped

**Test:** Asked "What is the size of the largest GPT-3 model in billions of parameters?" against the GPT-3 paper.

**Result:** The system answered "3B" — a hallucinated number, confidently cited to a page that did not contain it. The correct answer is 175B parameters, listed in the paper's Table 2.1.

**Root cause:** `PyPDFLoader` only extracts plain text from PDFs — it does not parse tables or figures. When inspecting the raw extracted text, the paper's text explicitly references "Figure 2.1" and "Table 2.1" by name, but the actual table/figure content is missing or garbled in the extracted output. The chunk retrieved by similarity search contained text *around* the table, not the table's actual data, so the LLM either said "I don't know" or pulled a nearby unrelated number and presented it as fact.

**Why this matters:** This is a common, underestimated failure mode in production RAG systems — any document with structured data (tables, charts, figures) will silently lose that information during plain-text extraction, leading to confident but wrong answers rather than honest "I don't know" responses.

### Fix implemented

Replaced `PyPDFLoader` with a custom `pdfplumber`-based loader that extracts tables separately and converts them into readable `cell | cell | cell` text with explicit `[TABLE]` markers, merged back into each page's content before chunking.

**Result:** Re-tested the same question — "What is the size of the largest GPT-3 model in billions of parameters?" — and the system now correctly answers **175B**, citing the page containing Table 2.1, where the extracted text shows the table structure clearly: `ModelName n_params n_layers d_model ... GPT-3...`

### Interview answer (updated)
"I stress-tested my RAG pipeline against three papers and found PyPDFLoader silently drops tables — when I asked about GPT-3's parameter count, the system hallucinated '3B' instead of 175B. I diagnosed this by inspecting the raw extracted text and confirmed tables were missing entirely. I fixed it by switching to pdfplumber, which extracts tables as structured data, and converted them into a readable pipe-delimited format merged back into the page text before chunking. After the fix, the same question correctly returned 175B with the right page citation. This taught me that production RAG systems need table-aware parsing — plain text extraction silently fails on structured data."

## Bonus — Embedding similarity demo

A standalone script (`similarity.py`) validating that the embedding model captures semantic meaning, not just keyword overlap. Tested cosine similarity across 7 sentence pairs.

| Comparison | Score | Interpretation |
|---|---|---|
| cat sat vs kitten rested | High (~0.75) | Different words, same meaning |
| cat sat vs quantum physics | Low (~0.12) | Unrelated meaning |
| attention mechanism vs multi-head attention | High (~0.68) | Same domain, related concept |
| attention mechanism vs quantum physics | Low (~0.15) | Unrelated meaning |

Confirms the retrieval layer of the RAG pipeline works on meaning rather than literal word matching — the core assumption behind why RAG retrieval is more powerful than keyword search.

# Day 2 — RAG Pipeline → FastAPI Production Service

A complete RAG pipeline that evolved from a command-line script into a tested FastAPI service. Ingests PDFs, stores them in a vector database, and answers questions grounded in the document with page citations — accessible via REST API.

## What it does
- Loads PDFs with table-aware extraction (pdfplumber)
- Splits into chunks (500 chars, 50 overlap) using RecursiveCharacterTextSplitter
- Embeds with all-MiniLM-L6-v2 (local, free, no API cost)
- Stores in ChromaDB, one collection per document (persisted to disk)
- Retrieves top-k chunks by cosine similarity
- Answers with Llama 3.1 via Groq, with page citations
- Exposed via FastAPI with `/ingest` and `/query` endpoints
- Tested with pytest covering success and failure paths

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
Python · FastAPI · Pydantic · LangChain · ChromaDB · HuggingFace Sentence Transformers · Groq API · Llama 3.1 8B · pdfplumber · pytest · NumPy

---

## Part 1 — Core RAG pipeline (`rag_pipeline.py`)

Standalone CLI script for direct testing.

```bash
python rag_pipeline.py attention_paper.pdf
```

### Test results
| Question | Answer | Hallucination? |
|---|---|---|
| What optimizer was used? | Adam β1=0.9, β2=0.98 (Page 6) | ✅ None |
| Who are the authors? | All 6 authors correct | ✅ None |
| What is multi-head attention? | Grounded from paper | ✅ None |

### Bonus — embedding similarity demo (`similarity.py`)
Validated that the embedding model captures semantic meaning, not just keyword overlap, using cosine similarity across 7 sentence pairs.

| Comparison | Score | Interpretation |
|---|---|---|
| cat sat vs kitten rested | High (~0.75) | Different words, same meaning |
| cat sat vs quantum physics | Low (~0.12) | Unrelated meaning |
| attention mechanism vs multi-head attention | High (~0.68) | Same domain, related concept |

Run it:
```bash
python similarity.py
```

---

## Part 2 — Multi-PDF stress test & bug fix

Tested the pipeline against 3 papers (Attention, BERT, GPT-3) to find failure modes.

### Bug found: PDF tables silently dropped
Asked "What is the size of the largest GPT-3 model in billions of parameters?" — the system answered **"3B"**, a hallucinated number confidently cited to the wrong page. The correct answer (175B) lives in a table that `PyPDFLoader` failed to extract.

**Root cause:** `PyPDFLoader` only extracts plain text — tables and figures are silently dropped or garbled. The retrieved chunk contained text *around* the missing table, not its data.

### Fix implemented
Replaced `PyPDFLoader` with a custom `pdfplumber`-based loader that extracts tables separately and converts them to readable `cell | cell | cell` text with `[TABLE]` markers, merged into page content before chunking.

**Result after fix:** Same question now correctly returns **175.0B**, citing the page containing Table 2.1.

### Interview answer
"I stress-tested my RAG pipeline against three papers and found PyPDFLoader silently drops tables — asking about GPT-3's parameter count returned a hallucinated '3B' instead of 175B. I diagnosed it by inspecting raw extracted text and confirmed the table was missing. I fixed it with pdfplumber's table extraction, converting tables into readable text merged into the page before chunking. The same question then correctly returned 175B with the right citation."

---

## Part 3 — FastAPI service (`api.py`)

Wrapped the pipeline in a REST API with request validation, error handling, and logging.

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

### Run it
```bash
uvicorn api:app --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive Swagger UI.

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

### Interview answer
"I wrapped my RAG pipeline in FastAPI with /ingest and /query endpoints. Pydantic models validate every request automatically — malformed requests get rejected with a 422 before they ever reach my business logic. I added structured logging to track every request and every LLM call, and proper HTTP status codes (404, 400, 500) so failures are predictable instead of crashing silently."

---

## Part 4 — Testing (`test_api.py`)

Wrote pytest tests covering both success and failure paths using FastAPI's `TestClient`.

| Test | What it checks |
|---|---|
| `test_health_check` | Health endpoint returns 200 and correct status |
| `test_ingest_missing_file_returns_404` | Ingesting a non-existent file fails gracefully |
| `test_ingest_non_pdf_returns_400` | Non-PDF files are rejected before processing |
| `test_query_missing_collection_returns_404` | Querying an un-ingested document returns a clear 404 |
| `test_query_missing_question_field_returns_422` | Pydantic rejects malformed requests automatically |
| `test_query_real_question_returns_valid_answer` | Full end-to-end: ingest → query → grounded answer with sources |

Run tests:
```bash
pytest test_api.py -v
```

### Why this matters
Most RAG demos only test the happy path. These tests explicitly cover failure modes — missing files, wrong file types, un-ingested collections, malformed requests — so the API fails predictably with proper status codes instead of crashing.

### Interview answer
"I wrote pytest tests using FastAPI's TestClient to cover both the happy path and failure cases — missing files, wrong file types, querying collections that don't exist, and malformed requests. Testing failure paths matters more than the happy path because that's where APIs actually break for real users."

---

## Files in this project
```
day2-rag-pipeline/
├── rag_pipeline.py     # standalone CLI RAG pipeline
├── similarity.py       # embedding similarity demo
├── api.py              # FastAPI service (/health, /ingest, /query)
├── test_api.py         # pytest test suite
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

for each of your 20 questions:
    1. search ChromaDB → get 3 paragraphs (retrieval)
    2. send those paragraphs + question to Llama → get an answer (generation)
    3. save: question, paragraphs used, answer given, correct answer

then hand ALL 20 of those records to RAGAS

RAGAS, using another LLM call as the "grader":
    for each of the 20 records:
        - check faithfulness: does the answer match the paragraphs?
        - check relevancy: does the answer match the question?
        - check recall: did the paragraphs actually contain the real answer?

finally: average all 20 scores per metric → that's your 3 final numbers

## RAGAS Evaluation

Built a 20-question evaluation set covering the paper end-to-end (authors, architecture, numbers, reasoning) and scored the RAG pipeline using RAGAS across 3 metrics.

### Results

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | 0.78 | Are answers grounded in retrieved context, not hallucinated? |
| Answer Relevancy | 1.00 | Do answers actually address the question asked? |
| Context Recall | 0.51 | Did retrieval pull back the chunks containing the real answer? |

### Bug found during evaluation: missing spaces in extracted text

Initial run scored faithfulness 0.45 and relevancy 0.48 — much lower than expected. Investigated by inspecting raw extracted text and found `pdfplumber`'s default text extraction was merging words together without spaces (e.g. "describedinsection3.2" instead of "described in section 3.2"), caused by default character-spacing tolerance being too loose for this PDF's font.

**Fix:** Added `x_tolerance=1, y_tolerance=3` to `extract_text()` calls, which restored correct word spacing.

**Result after fix:** Faithfulness jumped from 0.45 → 0.78, answer relevancy from 0.48 → 1.00. Confirms the LLM couldn't properly parse run-together text, causing both hallucination and irrelevant answers — once it could read clean text, it grounded answers correctly and stayed on-topic.

### Known remaining limitation: context recall (0.51)

Context recall did not improve with the spacing fix, since it measures retrieval quality, not generation quality. Inspecting retrieved chunks showed retrieval often pulls back near-duplicate chunks (due to chunk_overlap=50 combined with k=3), effectively wasting retrieval slots and missing some correct chunks roughly half the time.

**Next steps to improve (not yet implemented):** increase k to 5 for more retrieval attempts, or increase chunk_size to reduce duplication from overlap, or add a deduplication step before passing chunks to the LLM.

### Interview answer
"I built a 20-question eval set and scored my RAG pipeline with RAGAS across faithfulness, answer relevancy, and context recall. My first run scored low — 0.45 faithfulness — which led me to discover a second bug: pdfplumber's default text extraction was losing spaces between words due to character-spacing tolerance settings, making text hard for the LLM to parse correctly. After fixing the tolerance parameters, faithfulness jumped to 0.78 and relevancy hit a perfect 1.0. Context recall stayed around 0.51 though, which told me the remaining bottleneck is retrieval, not generation — likely duplicate chunks from my overlap settings wasting retrieval slots. That's a clear next optimization: increasing k or adjusting chunk size to reduce duplication."

## What I learned (updated)
- Evaluation isn't just a vanity metric — it surfaced a second, distinct bug (text-spacing) that manual testing had missed entirely
- Faithfulness and relevancy are generation-quality metrics; context recall is a retrieval-quality metric — they can move independently of each other
- pdfplumber's text extraction quality depends heavily on tolerance parameters, which vary by PDF font/layout
- A RAG system can score well on "did you answer correctly" while still having a real retrieval weakness hiding underneath



