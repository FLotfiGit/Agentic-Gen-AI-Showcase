"""Anomaly detection package for agentic time-series monitoring."""
from .agent import AnomalyAgent, AnomalyAgentConfig
from .detectors import Detector, ZScoreDetector, IQRDetector, StreamingThresholdDetector, DetectionResult
from .evaluation import confusion_matrix, precision_recall_f1
from .datasets import generate_spikes, generate_drift

__all__ = [
    "AnomalyAgent",
    "AnomalyAgentConfig",
    "Detector",
    "ZScoreDetector",
    "IQRDetector",
    "StreamingThresholdDetector",
    "DetectionResult",
    "confusion_matrix",
    "precision_recall_f1",
    "generate_spikes",
    "generate_drift",
]
