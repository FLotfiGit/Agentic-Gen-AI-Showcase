"""A small CLI runner to demonstrate the SimpleAgent planning loop.

This script uses `agents.agent_utils.SimpleAgent` with a local fallback LLM wrapper.
Set `OPENAI_API_KEY` to use an actual OpenAI call; otherwise the stub LLM is used.
"""
import os
import argparse
import json
from agents.agent_utils import SimpleAgent, stub_llm


def openai_llm(prompt: str) -> str:
    # Minimal optional OpenAI wrapper to show how to plug in a real model.
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


def main():
    parser = argparse.ArgumentParser(description="Run a tiny agent demo")
    parser.add_argument("--goal", default="Write a short summary about agentic AI")
    parser.add_argument("--max-steps", type=int, default=3)
    args = parser.parse_args()

    llm = openai_llm if os.getenv("OPENAI_API_KEY") else stub_llm
    agent = SimpleAgent(llm)
    res = agent.run(args.goal, max_steps=args.max_steps)
    print(json.dumps({"thoughts": [t.text for t in res.thoughts], "actions": [a.name for a in res.actions], "final": res.final}, indent=2))


if __name__ == "__main__":
    main()
