import numpy as np
from retrieval.faiss_index import build_embeddings, build_faiss_index, search_index


def test_build_embeddings_and_search():
    corpus = ["alpha", "beta", "gamma"]
    emb = build_embeddings(corpus)
    assert emb.shape[0] == 3
    idx = build_faiss_index(emb)
    q_emb = build_embeddings(["alpha"])[0]
    D, I = search_index(idx, q_emb, k=1)
    assert I[0] in (0,)
