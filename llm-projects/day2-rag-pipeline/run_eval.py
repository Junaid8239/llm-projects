import os
import sys
from unittest.mock import MagicMock
from dotenv import load_dotenv

# Patch missing module before ragas tries to import it
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from eval_set import eval_questions

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

load_dotenv("../.env")

# ── Load existing vectorstore for attention_paper ─────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db_attention_paper",
    embedding_function=embeddings
)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are an expert assistant answering questions about a document.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}

Answer:""")

# ── Run every question through the RAG pipeline ────────────
print(f"Running {len(eval_questions)} questions through RAG pipeline...\n")

results = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for i, item in enumerate(eval_questions):
    question = item["question"]
    ground_truth = item["ground_truth"]

    print(f"[{i+1}/{len(eval_questions)}] {question}")

    docs = vectorstore.similarity_search(question, k=3)
    contexts = [doc.page_content for doc in docs]
    context_str = "\n\n".join(contexts)

    chain = prompt | llm
    response = chain.invoke({"context": context_str, "question": question})

    results["question"].append(question)
    results["answer"].append(response.content)
    results["contexts"].append(contexts)
    results["ground_truth"].append(ground_truth)

print("\n✓ All questions answered. Running RAGAS evaluation...\n")

# ── Convert to RAGAS dataset format ─────────────────────────
dataset = Dataset.from_dict(results)

# ── Run RAGAS evaluation ────────────────────────────────────
score = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall],
    llm=llm,
    embeddings=embeddings,
)

print("\n" + "="*50)
print("RAGAS EVALUATION RESULTS")
print("="*50)
print(score)

# ── Save detailed results to CSV ────────────────────────────
df = score.to_pandas()
df.to_csv("ragas_eval_results.csv", index=False)
print("\n✓ Detailed results saved to ragas_eval_results.csv")