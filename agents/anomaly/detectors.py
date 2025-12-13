from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Iterable, Tuple
import math


@dataclass
class DetectionResult:
    indices: List[int]
    scores: List[float]
    threshold: float
    metadata: Dict[str, Any]


class Detector:
    def fit(self, series: Iterable[float]) -> None:
        pass

    def detect(self, series: Iterable[float]) -> DetectionResult:
        raise NotImplementedError


class ZScoreDetector(Detector):
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self._mean: Optional[float] = None
        self._std: Optional[float] = None

    def fit(self, series: Iterable[float]) -> None:
        data = list(series)
        if not data:
            self._mean, self._std = 0.0, 0.0
            return
        m = sum(data) / len(data)
        var = sum((x - m) ** 2 for x in data) / max(1, len(data) - 1)
        self._mean, self._std = m, math.sqrt(var)

    def detect(self, series: Iterable[float]) -> DetectionResult:
        data = list(series)
        if self._mean is None or self._std is None:
            self.fit(data)
        mean, std = self._mean or 0.0, self._std or 0.0
        scores = [(abs(x - mean) / std) if std > 0 else 0.0 for x in data]
        indices = [i for i, s in enumerate(scores) if s >= self.threshold]
        return DetectionResult(
            indices=indices,
            scores=scores,
            threshold=self.threshold,
            metadata={"method": "zscore", "mean": mean, "std": std},
        )


class IQRDetector(Detector):
    def __init__(self, k: float = 1.5):
        self.k = k
        self._q1: Optional[float] = None
        self._q3: Optional[float] = None

    @staticmethod
    def _percentile(sorted_vals: List[float], p: float) -> float:
        if not sorted_vals:
            return 0.0
        idx = (len(sorted_vals) - 1) * p
        lower = math.floor(idx)
        upper = math.ceil(idx)
        if lower == upper:
            return sorted_vals[int(idx)]
        return sorted_vals[lower] * (upper - idx) + sorted_vals[upper] * (idx - lower)

    def fit(self, series: Iterable[float]) -> None:
        data = sorted(list(series))
        self._q1 = self._percentile(data, 0.25)
        self._q3 = self._percentile(data, 0.75)

    def detect(self, series: Iterable[float]) -> DetectionResult:
        data = list(series)
        if self._q1 is None or self._q3 is None:
            self.fit(data)
        q1, q3 = self._q1 or 0.0, self._q3 or 0.0
        iqr = q3 - q1
        lower = q1 - self.k * iqr
        upper = q3 + self.k * iqr
        scores = [0.0] * len(data)
        indices = []
        for i, x in enumerate(data):
            if x < lower:
                indices.append(i)
                scores[i] = (lower - x) / (iqr if iqr > 0 else 1.0)
            elif x > upper:
                indices.append(i)
                scores[i] = (x - upper) / (iqr if iqr > 0 else 1.0)
        return DetectionResult(
            indices=indices,
            scores=scores,
            threshold=self.k,
            metadata={"method": "iqr", "q1": q1, "q3": q3, "iqr": iqr},
        )


class StreamingThresholdDetector(Detector):
    def __init__(self, window: int = 50, threshold: float = 3.0):
        self.window = window
        self.threshold = threshold
        self.buffer: List[float] = []

    def fit(self, series: Iterable[float]) -> None:
        # No offline fit, streaming-only
        pass

    def update(self, x: float) -> Tuple[bool, float]:
        # Score before adding to buffer
        if len(self.buffer) >= 3:
            m = sum(self.buffer) / len(self.buffer)
            var = sum((v - m) ** 2 for v in self.buffer) / max(1, len(self.buffer) - 1)
            std = math.sqrt(var)
            score = abs(x - m) / std if std > 0 else 0.0
        else:
            score = 0.0
        
        self.buffer.append(x)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        
        return score >= self.threshold, score

    def detect(self, series: Iterable[float]) -> DetectionResult:
        indices: List[int] = []
        scores: List[float] = []
        for i, x in enumerate(series):
            is_anom, s = self.update(float(x))
            scores.append(s)
            if is_anom:
                indices.append(i)
        return DetectionResult(
            indices=indices,
            scores=scores,
            threshold=self.threshold,
            metadata={"method": "streaming_threshold", "window": self.window},
        )
