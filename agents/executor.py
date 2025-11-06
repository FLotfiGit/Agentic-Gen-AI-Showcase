"""A tiny action executor for agent demos.

The executor simulates performing actions and returns deterministic results so tests
can run without side effects or external APIs.
"""
from typing import Dict, Any


def execute(action: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate executing an action dict and return a result dict.

    Expected action dict keys: {name: str, args: dict}
    """
    name = action.get("name", "noop")
    args = action.get("args", {}) or {}
    if name == "search":
        query = args.get("query", "")
        return {"status": "ok", "result": f"Simulated search results for '{query}'"}
    if name == "generate":
        prompt = args.get("prompt", "")
        return {"status": "ok", "result": f"Simulated generated text for prompt: {prompt[:80]}"}
    if name == "noop":
        return {"status": "ok", "result": f"No-op executed: {args.get('note', '')}"}
    # fallback
    return {"status": "ok", "result": f"Executed {name} with {args}"}


if __name__ == "__main__":
    print(execute({"name": "search", "args": {"query": "agentic AI"}}))
