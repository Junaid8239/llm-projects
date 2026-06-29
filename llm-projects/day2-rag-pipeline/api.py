import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv("../.env")

# ── Logging setup ──────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rag_api")

# ── App Configuration ──────────────────────────────────
app = FastAPI(title="Production RAG Pipeline API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Database & Model Initializations ───────────────────
# In production, these variables point to your cloud instance (e.g., Qdrant Cloud or a Railway Shared DB)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)

# Initialize the low-level client for administrative tasks (like dropping collections natively)
db_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

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


# ── Pydantic models ─────────────────────────────────────
class IngestResponse(BaseModel):
    collection_name: str
    pages_loaded: int
    chunks_created: int

class QueryRequest(BaseModel):
    collection_name: str = Field(..., description="Which database collection to query")
    question: str = Field(..., min_length=1, description="The question to ask")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of chunks to retrieve")

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


# ── Document Parsing Engine ─────────────────────────────
def load_pdf_with_tables(pdf_stream) -> list[Document]:
    """Parses a PDF completely in-memory using an open file stream."""
    pages = []
    with pdfplumber.open(pdf_stream) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
            tables = page.extract_tables()
            table_text = ""
            for t_idx, table in enumerate(tables):
                table_text += f"\n\n[TABLE {t_idx+1} on this page]\n"
                for row in table:
                    clean_row = [cell if cell else "" for cell in row]
                    table_text += " | ".join(clean_row) + "\n"
            pages.append(Document(
                page_content=text + table_text,
                metadata={"page": i}
            ))
    return pages


def process_and_ingest_stream(file_stream, collection_name: str) -> IngestResponse:
    """Core production pipeline: Core chunking and network-based DB injection."""
    logger.info(f"Starting database ingestion layer for collection: '{collection_name}'")
    
    pages = load_pdf_with_tables(file_stream)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    # NATIVE PRODUCTION WIPE: Recreate the collection over the network to avoid duplicates.
    # This completely bypasses any local OS file-locking mechanics.
    if db_client.collection_exists(collection_name):
        db_client.delete_collection(collection_name)
        logger.info(f"Natively dropped cloud collection '{collection_name}' to overwrite data clean.")

    # Explicitly create a clean collection parameters natively
    db_client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(size=384, distance=qdrant_models.Distance.COSINE)
    )

    # Stream vectors securely over the network to the standalone database
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name
    )

    logger.info(f"Successfully uploaded {len(chunks)} chunks to cloud instance for '{collection_name}'")
    return IngestResponse(
        collection_name=collection_name,
        pages_loaded=len(pages),
        chunks_created=len(chunks)
    )


# ── Endpoints ─────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "ok", "database_connected": db_client.get_collections() is not None}


@app.post("/ingest-upload", response_model=IngestResponse)
async def ingest_uploaded_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    collection_name = file.filename.lower().replace(".pdf", "").replace(" ", "_").replace("-", "_")
    
    try:
        # Read file completely in-memory as a byte stream - zero local disk allocation
        pdf_data = await file.read()
        import io
        pdf_stream = io.BytesIO(pdf_data)
        
        return process_and_ingest_stream(pdf_stream, collection_name)
    except Exception as e:
        logger.exception(f"Production ingestion failure for file: {file.filename}")
        raise HTTPException(status_code=500, detail=f"Database upload failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
def query_document(request: QueryRequest):
    logger.info(f"Querying cloud collection '{request.collection_name}'")

    if not db_client.collection_exists(request.collection_name):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{request.collection_name}' does not exist on the database cloud cluster."
        )

    try:
        # Initialize network-backed store pointer
        vectorstore = QdrantVectorStore(
            client=db_client,
            collection_name=request.collection_name,
            embeddings=embeddings
        )
        
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
        logger.exception(f"Query failed for cluster collection '{request.collection_name}'")
        raise HTTPException(status_code=500, detail=f"Cloud Query Failed: {str(e)}")