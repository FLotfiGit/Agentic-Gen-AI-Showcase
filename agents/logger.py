"""Agent run logger: append JSONL entries to outputs/agent_runs.jsonl

Each run writes a JSON object with: run_id, start_ts, end_ts, goal, thoughts, actions, results
"""
import json
import uuid
from pathlib import Path
from time import time
from typing import Any, Dict

OUT_PATH = Path("./outputs")
OUT_PATH.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUT_PATH / "agent_runs.jsonl"


def new_run_record(goal: str) -> Dict[str, Any]:
    return {"run_id": str(uuid.uuid4()), "goal": goal, "start_ts": time(), "end_ts": None, "thoughts": [], "actions": [], "results": []}


def finalize_run(record: Dict[str, Any]):
    record["end_ts"] = time()
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_thought(record: Dict[str, Any], thought: str):
    record.setdefault("thoughts", []).append({"text": thought, "ts": time()})


def append_action(record: Dict[str, Any], action: Dict[str, Any]):
    record.setdefault("actions", []).append({"action": action, "ts": time()})


def append_result(record: Dict[str, Any], result: Any):
    record.setdefault("results", []).append({"result": result, "ts": time()})


if __name__ == "__main__":
    r = new_run_record("Demo goal")
    append_thought(r, "Think about demo")
    append_action(r, {"name": "noop"})
    append_result(r, {"status": "ok"})
    finalize_run(r)
    print("Wrote to", LOG_FILE)
