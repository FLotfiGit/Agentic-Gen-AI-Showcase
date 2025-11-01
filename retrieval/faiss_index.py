"""A thin FAISS index wrapper for in-repo retrieval demos.

This module depends on `sentence_transformers` and `faiss` (faiss-cpu recommended).
It provides utilities to build an index from a list of texts and run similarity searches.
"""
from typing import List, Tuple
from pathlib import Path
import numpy as np


def build_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers not installed: " + str(e))
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return emb


def build_faiss_index(embeddings: np.ndarray):
    try:
        import faiss
    except Exception as e:
        raise RuntimeError("faiss not installed: " + str(e))
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index


def search_index(index, query_emb: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    D, I = index.search(query_emb.reshape(1, -1), k)
    return D[0], I[0]


def save_index(index, path: str):
    import faiss

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, path)


def load_index(path: str):
    import faiss

    return faiss.read_index(path)
