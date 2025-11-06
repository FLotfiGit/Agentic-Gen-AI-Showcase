"""Simple planner utilities for agentic workflows.

This module contains a tiny goal decomposition helper used by the SimpleAgent
for demonstration and tests. It's intentionally deterministic and dependency-free.
"""
from typing import List


def decompose_goal(goal: str, max_steps: int = 5) -> List[str]:
    """Create a small list of sub-goals (strings) from the high-level goal.

    Strategy (simple heuristics):
    - If the goal contains commas or 'and', split on those.
    - Otherwise, split by sentence punctuation.
    - If still one chunk, make synthetic subtasks: [research, synthesize, produce].
    """
    if not goal or not goal.strip():
        return []
    g = goal.strip()
    # split on commas or ' and '
    if "," in g or " and " in g.lower():
        parts = [p.strip() for p in g.replace(" and ", ",").split(",") if p.strip()]
    else:
        # naive sentence split
        parts = [p.strip() for p in g.replace("?", ".").replace("!", ".").split(".") if p.strip()]

    if len(parts) == 1:
        # synthesize simple 3-step plan
        return [f"Research: {g}", f"Synthesize findings about: {g}", f"Draft short summary for: {g}"][:max_steps]
    return parts[:max_steps]


if __name__ == "__main__":
    print(decompose_goal("Summarize agentic AI and retrieval-augmented generation"))
