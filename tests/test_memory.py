from agents.memory import SimpleMemory


def test_memory_add_and_retrieve():
    m = SimpleMemory()
    m.add("Learn about agents")
    m.add("Implement retrieval demo")
    res = m.retrieve("agents", k=2)
    assert len(res) >= 1


def test_memory_save_and_load(tmp_path):
    m = SimpleMemory()
    m.add("x")
    p = tmp_path / "mem.jsonl"
    m.save(str(p))
    m2 = SimpleMemory()
    m2.load(str(p))
    assert len(m2.all()) == 1
