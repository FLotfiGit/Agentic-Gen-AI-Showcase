#!/usr/bin/env bash
# Run the example CLI agent and the lightweight example workflow.
set -euo pipefail

mkdir -p outputs

echo "Running example_agent_workflow.py (deterministic stub LLM)"
python agents/example_agent_workflow.py || true

echo "Running CLI agent for a sample goal (stub LLM by default)"
python agents/cli_agent.py --goal "Summarize retrieval-augmented generation" --max-steps 3 > outputs/cli_agent_run.json || true

echo "Agent workflow finished; outputs are in ./outputs/"
