# Agent Quickstart

This file shows how to run the tiny agent and retrieval demos that live in this repository.

Run the CLI agent (uses stub LLM unless `OPENAI_API_KEY` is set):

```bash
python agents/run_agent.py --goal "Summarize recent work on agentic AI" --max-steps 3
```

Run the retriever demo (requires `sentence-transformers` and `faiss-cpu`):

```bash
python rag_systems/retriever_demo.py
```

Outputs will be saved to `./outputs/` where appropriate.
