"""High-level Retriever wrapper around sentence-transformers + FAISS helper.

Provides a simple API for adding texts, querying, and persisting index & embeddings.
"""
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np

from retrieval.faiss_index import build_embeddings, build_faiss_index, save_index, load_index


class Retriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._embeddings: Optional[np.ndarray] = None
        self._texts: List[str] = []
        self._index = None

    def add_texts(self, texts: List[str]):
        """Add texts to the in-memory corpus and (re)build embeddings and index."""
        if not texts:
            return
        self._texts.extend(texts)
        emb = build_embeddings(self._texts, model_name=self.model_name)
        self._embeddings = emb
        self._index = build_faiss_index(emb)

    def query(self, q: str, k: int = 3) -> List[Tuple[float, str]]:
        """Return list of (distance, text) for top-k nearest neighbors."""
        if self._index is None or self._embeddings is None:
            raise RuntimeError("Index is empty. Call add_texts() first.")
        q_emb = build_embeddings([q], model_name=self.model_name)[0]
        D, I = self._index.search(q_emb.reshape(1, -1), k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            results.append((float(dist), self._texts[int(idx)]))
        return results

    def save(self, path: str):
        """Save the FAISS index and the plaintext corpus to disk.

        Saves: <path>.idx (faiss index), <path>.txt (one doc per line)
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # save index
        if self._index is None:
            raise RuntimeError("No index to save")
        save_index(self._index, str(p.with_suffix(".idx")))
        # save corpus
        with open(p.with_suffix(".txt"), "w", encoding="utf-8") as f:
            for t in self._texts:
                f.write(t.replace("\n", " ") + "\n")

    def load(self, path: str):
        """Load an existing faiss index and corpus file from disk.

        Expects: <path>.idx and <path>.txt
        """
        p = Path(path)
        idx_path = p.with_suffix(".idx")
        txt_path = p.with_suffix(".txt")
        if not idx_path.exists() or not txt_path.exists():
            raise FileNotFoundError("Index or corpus file missing")
        self._index = load_index(str(idx_path))
        with open(txt_path, "r", encoding="utf-8") as f:
            self._texts = [l.strip() for l in f.readlines() if l.strip()]
