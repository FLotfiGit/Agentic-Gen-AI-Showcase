"""A small, test-friendly in-memory memory store for agents.

This store is intentionally simple: it records timestamped text entries and provides
basic retrieval by substring or token overlap (no heavy dependencies).
"""
from typing import List, Dict, Any
from dataclasses import dataclass, field
import time
import json
from pathlib import Path


@dataclass
class MemoryItem:
    text: str
    timestamp: float = field(default_factory=lambda: time.time())


class SimpleMemory:
    """Append-only memory with basic retrieval.

    Methods:
    - add(text)
    - retrieve(query, k=3) -> list of MemoryItem (best matches by substring/token overlap)
    - save(path), load(path)
    """

    def __init__(self):
        self._items: List[MemoryItem] = []

    def add(self, text: str):
        self._items.append(MemoryItem(text=text))

    def all(self) -> List[MemoryItem]:
        return list(self._items)

    def retrieve(self, query: str, k: int = 3) -> List[MemoryItem]:
        if not query:
            # return most recent
            return list(sorted(self._items, key=lambda i: i.timestamp, reverse=True))[:k]
        q = query.lower()
        # score by simple token overlap and substring presence
        def score(item: MemoryItem):
            t = item.text.lower()
            s = 0
            if q in t:
                s += 10
            qtokens = set(q.split())
            ttokens = set(t.split())
            s += len(qtokens & ttokens)
            # recency tie-breaker
            s += max(0, 1 - ((time.time() - item.timestamp) / (60 * 60 * 24)))
            return s

        scored = sorted(self._items, key=lambda it: score(it), reverse=True)
        return scored[:k]

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for it in self._items:
                f.write(json.dumps({"text": it.text, "timestamp": it.timestamp}) + "\n")

    def load(self, path: str):
        p = Path(path)
        if not p.exists():
            return
        self._items = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                j = json.loads(line)
                self._items.append(MemoryItem(text=j.get("text", ""), timestamp=j.get("timestamp", time.time())))


if __name__ == "__main__":
    m = SimpleMemory()
    m.add("Read paper on agentic AI")
    m.add("Implement retrieval demo")
    print([i.text for i in m.retrieve("agentic")])
