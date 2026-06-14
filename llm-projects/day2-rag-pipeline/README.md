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
