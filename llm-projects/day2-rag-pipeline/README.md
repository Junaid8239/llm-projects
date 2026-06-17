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
