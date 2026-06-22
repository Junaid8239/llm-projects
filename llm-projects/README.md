# LLM Projects — GenAI Engineering Portfolio
Building in public: SWE → GenAI Engineer in 31 days.

## Live Demo
**Day 2 RAG Pipeline is deployed and live:** [llm-projects-production.up.railway.app/docs](https://llm-projects-production.up.railway.app/docs)
Try it: `POST /ingest` a PDF, then `POST /query` to ask grounded questions with page citations.

## Projects

### Day 1 — Chatbot with memory
LangChain + Groq (Llama 3.1). Conversational chatbot with message history.
→ [`day1-hello-llm/`](./day1-hello-llm)

### Day 2 — RAG Pipeline (PDF → ChromaDB → Q&A with citations) — 🟢 LIVE
PDF → table-aware chunking → local embeddings → ChromaDB → Llama 3.1 via Groq → grounded answers with page citations.
- **FastAPI service** with `/ingest`, `/query`, `/health` endpoints
- **Evaluated with RAGAS:** 0.78 faithfulness · 1.00 answer relevancy · 0.51 context recall
- **Tested:** 6 pytest tests covering success and failure paths
- **Containerized:** Docker + docker-compose with persistent ChromaDB volumes
- **Deployed:** live on Railway with a persistent volume (verified to survive restarts) — [public URL above](https://llm-projects-production.up.railway.app/docs)
- **CI:** GitHub Actions runs the full test suite on every push
→ [`day2-rag-pipeline/`](./day2-rag-pipeline) — see its README for the full build/debug story, including two real bugs found and fixed (hallucinated table data, missing text-spacing) and how the deployment was verified.

## Stack
Python · LangChain · Groq (Llama 3.1) · ChromaDB · HuggingFace Sentence Transformers · FastAPI · pdfplumber · pytest · Docker · Railway · GitHub Actions