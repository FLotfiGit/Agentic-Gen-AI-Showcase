from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Callable, Optional

from .detectors import Detector, DetectionResult, ZScoreDetector
from .features import build_basic_features


Callback = Callable[[Dict[str, Any]], None]


@dataclass
class AnomalyAgentConfig:
    window: int = 10
    detector_threshold: float = 3.0


class AnomalyAgent:
    def __init__(
        self,
        detector: Optional[Detector] = None,
        config: Optional[AnomalyAgentConfig] = None,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        self.config = config or AnomalyAgentConfig()
        self.detector = detector or ZScoreDetector(threshold=self.config.detector_threshold)
        self.callbacks = callbacks or []

    def on_event(self, payload: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            try:
                cb(payload)
            except Exception:
                # keep agent resilient
                pass

    def run(self, series: Iterable[float]) -> Dict[str, Any]:
        vals = list(series)
        feat_res = build_basic_features(vals, window=self.config.window)
        det_res = self.detector.detect(vals)

        result = {
            "indices": det_res.indices,
            "scores": det_res.scores,
            "threshold": det_res.threshold,
            "detector": det_res.metadata,
            "features": feat_res.metadata,
        }
        self.on_event({"type": "anomaly_result", "data": result})
        return result
