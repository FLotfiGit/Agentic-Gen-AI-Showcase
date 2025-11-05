import tempfile
from retrieval.retriever import Retriever


def test_retriever_add_and_query():
    texts = [
        "Agentic AI composes plans by reflecting on its actions.",
        "Retrieval augments language models with external context.",
        "FAISS provides fast similarity search for dense vectors.",
    ]
    r = Retriever()
    r.add_texts(texts)
    res = r.query("How does retrieval help generation?", k=2)
    assert len(res) == 2
    # distances should be finite floats and returned texts should be from the corpus
    assert all(isinstance(d, float) for d, _ in res)
    assert all(t in texts for _, t in res)


def test_retriever_save_and_load(tmp_path):
    texts = ["alpha","beta"]
    r = Retriever()
    r.add_texts(texts)
    base = tmp_path / "myidx"
    r.save(str(base))
    # create new retriever and load
    r2 = Retriever()
    r2.load(str(base))
    res = r2.query("alpha", k=1)
    assert len(res) == 1
