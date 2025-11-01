from agents.agent_utils import SimpleAgent, stub_llm, Thought, Action


def test_stub_llm_returns_three_thoughts_for_summary():
    agent = SimpleAgent(stub_llm)
    thoughts = agent.plan("Summarize this document")
    assert len(thoughts) >= 1
    assert all(isinstance(t, Thought) for t in thoughts)


def test_act_produces_known_action_names():
    agent = SimpleAgent(stub_llm)
    t = Thought(text="Search for recent papers on agentic AI")
    a = agent.act(t)
    assert isinstance(a, Action)
    assert a.name in ("search", "noop")
