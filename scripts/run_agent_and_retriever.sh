#!/usr/bin/env bash
# Simple runner for local demos: runs agent and retriever demos and saves outputs to ./outputs/
set -euo pipefail

python agents/run_agent.py --goal "List key ideas of retrieval-augmented generation" --max-steps 2 > outputs/agent_demo.json || true
python rag_systems/retriever_demo.py || true

echo "Demos completed. Check ./outputs/ for artifacts."
