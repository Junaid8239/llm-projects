from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate



load_dotenv("../.env")  # one level up

# ── 1. LOAD ──────────────────────────────────────────────
print("Loading PDF...")
loader = PyPDFLoader("attention_paper.pdf")
pages = loader.load()
print(f"✓ Loaded {len(pages)} pages")

# ── 2. CHUNK ─────────────────────────────────────────────
print("\nChunking...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(pages)
print(f"✓ Created {len(chunks)} chunks")

# ── 3. EMBED + STORE ─────────────────────────────────────
print("\nLoading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

import os
if os.path.exists("./chroma_db"):
    print("✓ ChromaDB found on disk — loading existing index")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
else:
    print("Embedding and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"✓ Stored {len(chunks)} chunks in ChromaDB")

# ── 4. LLM ───────────────────────────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are an expert on the "Attention Is All You Need" paper.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the document."
Always cite which page you found the answer on.

Context:
{context}

Question: {question}

Answer:""")

# ── 5. Q&A LOOP ──────────────────────────────────────────
print("\n" + "─"*50)
print("📄 RAG Q&A — Attention Is All You Need")
print("─"*50)
print("Type your question. Type 'quit' to exit.\n")

while True:
    question = input("You: ").strip()

    if question.lower() == "quit":
        break
    if not question:
        continue

    # Retrieve top 3 relevant chunks
    docs = vectorstore.similarity_search(question, k=3)

    # Build context with page numbers
    context_parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[Page {page}]: {doc.page_content}")
    context = "\n\n".join(context_parts)

    # Generate answer
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })

    print(f"\nBot: {response.content}")
    print("\n📎 Retrieved from:")
    for doc in docs:
        print(f"  Page {doc.metadata.get('page','?')}: {doc.page_content[:80]}...")
    print()