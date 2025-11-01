"""Simple agent utilities for plan-act-revise loops.

This module provides tiny dataclasses and a driver to run a planning-act cycle.
It is intentionally minimal and dependency-free so tests can run without API keys.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import json
import time


@dataclass
class Thought:
    text: str
    confidence: float = 1.0


@dataclass
class Action:
    name: str
    args: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    thoughts: List[Thought]
    actions: List[Action]
    final: Optional[str] = None


class SimpleAgent:
    """A toy agent that runs a plan->act loop using a provided LLM-like callable.

    The llm callable should accept a prompt (str) and return a string response. For
    unit tests a stub function can be provided.
    """

    def __init__(self, llm_callable, name: str = "simple-agent"):
        self.name = name
        self.llm = llm_callable

    def plan(self, goal: str) -> List[Thought]:
        # Very small prompt to the llm: return up to 3 thoughts separated by \n
        prompt = f"You are an agent planning to: {goal}\nList 1-3 short thoughts (one per line) describing steps or considerations."
        resp = self.llm(prompt)
        thoughts = [Thought(text=t.strip()) for t in resp.splitlines() if t.strip()][:3]
        return thoughts

    def act(self, thought: Thought) -> Action:
        # Try to convert a thought into a simple action name
        text = thought.text.lower()
        if "search" in text or "lookup" in text or "find" in text:
            return Action(name="search", args={"query": thought.text})
        if "write" in text or "compose" in text or "generate" in text:
            return Action(name="generate", args={"prompt": thought.text})
        # fallback
        return Action(name="noop", args={"note": thought.text})

    def run(self, goal: str, max_steps: int = 3) -> AgentResult:
        thoughts = self.plan(goal)
        actions = []
        for t in thoughts[:max_steps]:
            act = self.act(t)
            actions.append(act)
        final = f"Completed {len(actions)} actions towards: {goal}"
        return AgentResult(thoughts=thoughts, actions=actions, final=final)


def stub_llm(prompt: str) -> str:
    """A deterministic small stub used for tests and local demos."""
    # echo back a predictable set of thoughts depending on keywords
    if "summarize" in prompt.lower():
        return "Read source\nExtract key points\nWrite summary"
    return "Search web for background\nRead top result\nWrite a short summary"


if __name__ == "__main__":
    # Simple CLI demo
    agent = SimpleAgent(stub_llm)
    res = agent.run("Write a short summary about agentic AI", max_steps=3)
    print(json.dumps({"thoughts": [t.text for t in res.thoughts], "actions": [a.name for a in res.actions], "final": res.final}, indent=2))
