# Agent Design (lightweight)

This document describes the small, test-friendly agent architecture used in the repository.

Components

- planner (`agents/planner.py`): deterministic goal decomposition helper. Splits goals into subtasks.
- agent (`agents/agent_utils.py`): `SimpleAgent` wraps an `llm_callable` and turns thoughts into actions.
- executor (`agents/executor.py`): deterministic action executor that simulates `search` and `generate` actions.
- memory (`agents/memory.py`): `SimpleMemory` append-only store with simple retrieval.
- logger (`agents/logger.py`): JSONL logger that writes a trace per run in `outputs/agent_runs.jsonl`.

Data flow

1. User provides a high-level goal.
2. The planner decomposes the goal into 1..N sub-goals.
3. For each sub-goal the agent produces thoughts/actions (via `SimpleAgent` + stub LLM by default).
4. Actions are executed via the `executor` and results are stored in memory.
5. The logger records thoughts, actions, results, and timestamps to `agent_runs.jsonl`.

Design choices

- No external services by default: everything works with the included `stub_llm` and deterministic executors so tests are fast and reproducible.
- Minimal, well-documented interfaces to swap in real LLMs, search tools, or databases later.

Running an example

```bash
# run the example workflow (produces outputs/agent_runs.jsonl)
python agents/example_agent_workflow.py
```

Next steps

- Integrate real LLMs behind `llm_callable` in `agents/agent_utils.py`.
- Add richer memory retrieval (embedding-based) and persistent stores (sqlite, vector DB).
- Add metrics and visualizations for agent traces.
