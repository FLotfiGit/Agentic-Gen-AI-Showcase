"""Advanced CLI agent that wires planner, memory, executor, and logger.

Usage:
  python agents/cli_agent.py --goal "Write a summary about RAG" --max-steps 3

Behavior:
  - Uses OpenAI ChatCompletion when OPENAI_API_KEY is set, otherwise falls back to stub_llm
  - Runs planner.decompose_goal, converts to actions via SimpleAgent, executes actions,
    stores results in memory, and logs the run to outputs/agent_runs.jsonl
"""
import os
import argparse
import json
from agents.planner import decompose_goal
from agents.agent_utils import SimpleAgent, stub_llm
from agents.executor import execute
from agents.memory import SimpleMemory
from agents.logger import new_run_record, append_thought, append_action, append_result, finalize_run


def openai_llm(prompt: str) -> str:
    try:
        import openai
    except Exception:
        return stub_llm(prompt)
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return stub_llm(prompt)
    openai.api_key = key
    resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], max_tokens=150, temperature=0.2)
    return resp.choices[0].message.content


def run_agent_cli(goal: str, max_steps: int = 3, use_openai: bool = False):
    llm = openai_llm if use_openai and os.getenv("OPENAI_API_KEY") else stub_llm
    agent = SimpleAgent(llm)
    mem = SimpleMemory()
    record = new_run_record(goal)

    steps = decompose_goal(goal, max_steps=max_steps)
    for s in steps:
        append_thought(record, s)
        thought_obj = type("T", (), {"text": s})
        action = agent.act(thought_obj)
        append_action(record, {"name": action.name, "args": action.args})
        res = execute({"name": action.name, "args": action.args})
        append_result(record, res)
        mem.add(json.dumps({"thought": s, "action": action.name, "result": res}))

    # simple reflection: store top memories
    recalls = mem.retrieve(goal, k=2)
    for r in recalls:
        append_thought(record, f"Memory recall: {r.text}")

    finalize_run(record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", required=True)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--use-openai", action="store_true", help="Attempt to use OpenAI if OPENAI_API_KEY is set")
    args = parser.parse_args()
    rec = run_agent_cli(args.goal, max_steps=args.max_steps, use_openai=args.use_openai)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
