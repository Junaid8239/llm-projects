from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

sentences = [
    "The cat sat on the mat",
    "A kitten rested on the rug",
    "Quantum physics studies subatomic particles",
    "The attention mechanism relates positions in a sequence",
    "Multi-head attention allows attending to different representation spaces",
    "I love eating pizza for dinner",
    "Neural networks learn from data",
    "The Transformer model uses self-attention",
    "Python is a programming language",
    "The model was trained using Adam optimizer",
]

print("Embedding 10 sentences...")
vectors = embeddings.embed_documents(sentences)
vectors = np.array(vectors)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\n── Similarity scores ──\n")

pairs = [
    (0, 1, "cat sat vs kitten rested"),
    (0, 2, "cat sat vs quantum physics"),
    (3, 4, "attention mechanism vs multi-head attention"),
    (3, 7, "attention mechanism vs Transformer model"),
    (3, 2, "attention mechanism vs quantum physics"),
    (6, 7, "neural networks vs Transformer model"),
    (0, 5, "cat sat vs pizza dinner"),
]

for i, j, label in pairs:
    score = cosine_similarity(vectors[i], vectors[j])
    bar = "█" * int(score * 30)
    print(f"{label}")
    print(f"  {score:.3f}  {bar}\n")