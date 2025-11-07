"""Example agentic workflow: plan -> act -> store in memory -> reflect -> log

This script composes the pieces (planner, agent_utils, executor, memory, logger)
to show a simple run that requires no external APIs.
"""
from agents.agent_utils import SimpleAgent, stub_llm
from agents.planner import decompose_goal
from agents.executor import execute
from agents.memory import SimpleMemory
from agents.logger import new_run_record, append_thought, append_action, append_result, finalize_run
import json


def run_example(goal: str = "Summarize agentic AI principles", max_steps: int = 3):
    record = new_run_record(goal)
    agent = SimpleAgent(stub_llm)
    mem = SimpleMemory()

    # Plan
    steps = decompose_goal(goal, max_steps=max_steps)
    for s in steps:
        append_thought(record, s)
        # Agent turns thought into action
        thought_obj = type("T", (), {"text": s})
        action = agent.act(thought_obj)
        append_action(record, {"name": action.name, "args": action.args})
        # Execute
        res = execute({"name": action.name, "args": action.args})
        append_result(record, res)
        # Store result in memory
        mem.add(json.dumps({"thought": s, "action": action.name, "result": res}))

    # Reflect (very small): retrieve memory and append as final thought
    mem_items = mem.retrieve(goal, k=2)
    for mi in mem_items:
        append_thought(record, f"Memory recall: {mi.text}")

    finalize_run(record)
    return record


if __name__ == "__main__":
    rec = run_example()
    print(json.dumps(rec, indent=2))
