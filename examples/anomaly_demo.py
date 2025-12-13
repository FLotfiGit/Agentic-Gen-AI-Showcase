from agents.anomaly.agent import AnomalyAgent, AnomalyAgentConfig
from agents.anomaly.detectors import ZScoreDetector
from agents.anomaly.datasets import generate_spikes
from agents.anomaly.evaluation import precision_recall_f1


def main():
    series, labels = generate_spikes(n=300, spike_rate=0.03)
    agent = AnomalyAgent(detector=ZScoreDetector(threshold=2.5), config=AnomalyAgentConfig(window=5))
    res = agent.run(series)
    metrics = precision_recall_f1(res["indices"], labels)
    print("indices:", res["indices"][:10])
    print("metrics:", metrics)


if __name__ == "__main__":
    main()
