import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv("../.env")

# ── Logging setup ──────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rag_api")

# ── App + globals ──────────────────────────────────────
app = FastAPI(title="RAG Pipeline API", version="1.0")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are an expert assistant answering questions about a document.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the document."
Always cite which page you found the answer on.

Context:
{context}

Question: {question}

Answer:""")

vectorstores = {}  # cache loaded vectorstores per collection


# ── Pydantic models ─────────────────────────────────────
class IngestRequest(BaseModel):
    pdf_path: str = Field(..., description="Path to the PDF file on disk")

class IngestResponse(BaseModel):
    collection_name: str
    pages_loaded: int
    chunks_created: int

class QueryRequest(BaseModel):
    collection_name: str = Field(..., description="Which ingested document to query")
    question: str = Field(..., min_length=1, description="The question to ask")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of chunks to retrieve")

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


# ── Helper functions ─────────────────────────────────────
def load_pdf_with_tables(pdf_path: str):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables()
            table_text = ""
            for t_idx, table in enumerate(tables):
                table_text += f"\n\n[TABLE {t_idx+1} on this page]\n"
                for row in table:
                    clean_row = [cell if cell else "" for cell in row]
                    table_text += " | ".join(clean_row) + "\n"
            pages.append(Document(
                page_content=text + table_text,
                metadata={"page": i, "source": pdf_path}
            ))
    return pages


# ── Endpoints ─────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest_document(request: IngestRequest):
    if not os.path.exists(request.pdf_path):
        logger.error(f"File not found: {request.pdf_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {request.pdf_path}")

    if not request.pdf_path.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    collection_name = os.path.basename(request.pdf_path).replace(".pdf", "").replace(" ", "_")
    logger.info(f"Ingesting {request.pdf_path} as collection '{collection_name}'")

    try:
        pages = load_pdf_with_tables(request.pdf_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)

        persist_dir = f"./chroma_db_{collection_name}"
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        vectorstores[collection_name] = vectorstore

        logger.info(f"Ingested {len(pages)} pages, {len(chunks)} chunks for '{collection_name}'")
        return IngestResponse(
            collection_name=collection_name,
            pages_loaded=len(pages),
            chunks_created=len(chunks)
        )
    except Exception as e:
        logger.exception(f"Ingestion failed for {request.pdf_path}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
def query_document(request: QueryRequest):
    persist_dir = f"./chroma_db_{request.collection_name}"

    if request.collection_name not in vectorstores:
        if os.path.exists(persist_dir):
            vectorstores[request.collection_name] = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{request.collection_name}' not found. Ingest it first via /ingest"
            )

    vectorstore = vectorstores[request.collection_name]
    logger.info(f"Querying '{request.collection_name}': {request.question}")

    try:
        docs = vectorstore.similarity_search(request.question, k=request.top_k)

        context_parts = []
        sources = []
        for doc in docs:
            page = doc.metadata.get("page", "?")
            context_parts.append(f"[Page {page}]: {doc.page_content}")
            sources.append(f"Page {page}")
        context = "\n\n".join(context_parts)

        chain = prompt | llm
        response = chain.invoke({"context": context, "question": request.question})

        return QueryResponse(answer=response.content, sources=sources)
    except Exception as e:
        logger.exception(f"Query failed for '{request.collection_name}'")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")