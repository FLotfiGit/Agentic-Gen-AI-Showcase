from agents.planner import decompose_goal


def test_decompose_empty_goal():
    assert decompose_goal("") == []


def test_decompose_commas_and_and():
    g = "Collect papers, read them and write summary"
    parts = decompose_goal(g, max_steps=5)
    assert isinstance(parts, list)
    assert len(parts) >= 2


def test_decompose_single_synthesized():
    g = "Summarize agentic AI"
    parts = decompose_goal(g, max_steps=3)
    assert len(parts) >= 1
