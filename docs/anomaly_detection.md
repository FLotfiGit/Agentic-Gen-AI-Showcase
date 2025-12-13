# Anomaly Detection Agent

This module provides a lightweight, agentic anomaly detection pipeline for numeric time-series.

## Components
- Detectors: Z-score, IQR, Streaming threshold
- Features: Rolling mean/std, normalization, composite features
- Agent: Orchestrates features -> detection -> callbacks
- Datasets: Synthetic spikes and drift with labels
- Evaluation: Confusion matrix, precision/recall/F1
- CLI: `cli/anomaly_detect.py` for quick runs

## Quickstart
```bash
python cli/anomaly_detect.py --window 10 --threshold 2.5 --input 0,0,0,10,0,0
```

## Python Usage
```python
from agents.anomaly.agent import AnomalyAgent, AnomalyAgentConfig
from agents.anomaly.detectors import ZScoreDetector

series = [0,0,0,10,0,0]
agent = AnomalyAgent(detector=ZScoreDetector(threshold=2.0), config=AnomalyAgentConfig(window=3))
res = agent.run(series)
print(res["indices"])  # [3]
```
