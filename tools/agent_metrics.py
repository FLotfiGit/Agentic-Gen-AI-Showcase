"""Compute simple metrics from outputs/agent_runs.jsonl

Usage:
  python tools/agent_metrics.py

Outputs counts and average durations, thoughts/actions per run.
"""
import json
from pathlib import Path
from statistics import mean


def read_runs(path: str = "outputs/agent_runs.jsonl"):
    p = Path(path)
    if not p.exists():
        print("No runs found at", p)
        return []
    runs = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            runs.append(json.loads(line))
    return runs


def summarize(runs):
    if not runs:
        print("No runs to summarize")
        return
    durations = [r.get("end_ts", 0) - r.get("start_ts", 0) for r in runs]
    thoughts_counts = [len(r.get("thoughts", [])) for r in runs]
    actions_counts = [len(r.get("actions", [])) for r in runs]
    print(f"Total runs: {len(runs)}")
    print(f"Avg duration (s): {mean(durations):.3f}")
    print(f"Avg thoughts per run: {mean(thoughts_counts):.2f}")
    print(f"Avg actions per run: {mean(actions_counts):.2f}")


if __name__ == "__main__":
    runs = read_runs()
    summarize(runs)
