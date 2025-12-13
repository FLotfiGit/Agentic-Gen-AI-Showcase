#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import List

from agents.anomaly.agent import AnomalyAgent, AnomalyAgentConfig
from agents.anomaly.detectors import ZScoreDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run anomaly detection over numeric series")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--threshold", type=float, default=3.0)
    p.add_argument("--input", type=str, help="Comma-separated numbers or path to JSON list")
    return p.parse_args()


def load_series(input_arg: str) -> List[float]:
    if input_arg.endswith(".json"):
        with open(input_arg, "r") as f:
            data = json.load(f)
        return [float(x) for x in data]
    return [float(x) for x in input_arg.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    series = load_series(args.input)

    agent = AnomalyAgent(
        detector=ZScoreDetector(threshold=args.threshold),
        config=AnomalyAgentConfig(window=args.window),
    )
    res = agent.run(series)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
