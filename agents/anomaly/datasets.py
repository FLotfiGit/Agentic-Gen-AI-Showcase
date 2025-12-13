from __future__ import annotations

from typing import List, Tuple
import random


def generate_spikes(n: int = 500, spike_rate: float = 0.02, noise: float = 0.1) -> Tuple[List[float], List[int]]:
    series: List[float] = []
    labels: List[int] = []
    for i in range(n):
        base = random.random()
        val = base + random.uniform(-noise, noise)
        if random.random() < spike_rate:
            val += random.uniform(3.0, 6.0)
            labels.append(1)
        else:
            labels.append(0)
        series.append(val)
    return series, labels


def generate_drift(n: int = 500, drift: float = 0.005, noise: float = 0.1) -> Tuple[List[float], List[int]]:
    series: List[float] = []
    labels: List[int] = []
    acc = 0.0
    for i in range(n):
        acc += drift
        val = acc + random.uniform(-noise, noise)
        # occasional spike in drifted series
        if random.random() < 0.01:
            val += random.uniform(2.0, 4.0)
            labels.append(1)
        else:
            labels.append(0)
        series.append(val)
    return series, labels
