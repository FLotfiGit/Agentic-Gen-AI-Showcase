from agents.anomaly.agent import AnomalyAgent, AnomalyAgentConfig
from agents.anomaly.detectors import ZScoreDetector


def test_agent_runs_and_returns_indices():
    agent = AnomalyAgent(detector=ZScoreDetector(threshold=1.5), config=AnomalyAgentConfig(window=3))
    res = agent.run([0, 0, 0, 10, 0])
    assert "indices" in res and isinstance(res["indices"], list)
    assert 3 in res["indices"]
