from __future__ import annotations

from dataclasses import dataclass
from typing import List, Iterable, Tuple, Dict, Any
import math


@dataclass
class FeatureResult:
    features: List[float]
    metadata: Dict[str, Any]


def rolling_mean(series: Iterable[float], window: int) -> List[float]:
    vals = list(series)
    out: List[float] = []
    s = 0.0
    for i, v in enumerate(vals):
        s += v
        if i >= window:
            s -= vals[i - window]
        out.append(s / min(window, i + 1))
    return out


def rolling_std(series: Iterable[float], window: int) -> List[float]:
    vals = list(series)
    out: List[float] = []
    buf: List[float] = []
    for v in vals:
        buf.append(v)
        if len(buf) > window:
            buf.pop(0)
        m = sum(buf) / len(buf)
        var = sum((x - m) ** 2 for x in buf) / max(1, len(buf) - 1)
        out.append(math.sqrt(var))
    return out


def normalize_minmax(series: Iterable[float]) -> List[float]:
    vals = list(series)
    if not vals:
        return []
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1.0
    return [(v - mn) / rng for v in vals]


def window_view(series: Iterable[float], window: int) -> List[List[float]]:
    vals = list(series)
    out: List[List[float]] = []
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        out.append(vals[start : i + 1])
    return out


def build_basic_features(series: Iterable[float], window: int = 10) -> FeatureResult:
    vals = list(series)
    mean_feat = rolling_mean(vals, window)
    std_feat = rolling_std(vals, window)
    norm_vals = normalize_minmax(vals)
    # simple composite feature: z-like normalized deviation
    comp = []
    for i, v in enumerate(vals):
        s = std_feat[i]
        m = mean_feat[i]
        comp.append((abs(v - m) / s) if s > 0 else 0.0)
    feat = [0.5 * a + 0.5 * b for a, b in zip(comp, norm_vals)]
    return FeatureResult(
        features=feat,
        metadata={
            "method": "basic_features",
            "window": window,
            "components": ["rolling_mean", "rolling_std", "minmax", "composite"],
        },
    )
