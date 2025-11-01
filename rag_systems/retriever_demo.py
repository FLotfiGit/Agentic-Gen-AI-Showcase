"""A simple demonstration showing how to embed a small corpus, build a FAISS index and query it."""
from pathlib import Path
from retrieval.faiss_index import build_embeddings, build_faiss_index, search_index, save_index


def demo():
    corpus = [
        "Agentic AI composes plans by reflecting on its actions.",
        "Retrieval-augmented generation uses a vector index to find relevant context.",
        "FAISS is a fast similarity search library for embeddings on CPU.",
    ]
    print("Building embeddings for corpus of size", len(corpus))
    emb = build_embeddings(corpus)
    idx = build_faiss_index(emb)
    q = "How does retrieval help generation?"
    q_emb = build_embeddings([q])[0]
    D, I = search_index(idx, q_emb, k=2)
    print("Query:", q)
    for dist, i in zip(D, I):
        print(f"- (dist={dist:.4f}) {corpus[i]}")
    # save index for later
    out = Path("outputs/faiss_index.idx")
    save_index(idx, str(out))
    print("Saved index to", out)


if __name__ == "__main__":
    demo()
