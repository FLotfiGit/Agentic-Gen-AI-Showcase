SHELL := /bin/bash

.PHONY: setup notebooks test clean

setup:
	bash scripts/setup.sh

notebooks:
	python - <<'PY'
import os
print('Notebooks available:')
for root, _, files in os.walk('.'):
    for f in files:
        if f.endswith('.ipynb'):
            print(os.path.join(root, f))
PY

test:
	python - <<'PY'
import json, os
ok=True
for d in ['agents','rag_systems','multimodal','generative_models','outputs']:
    if not os.path.exists(d):
        print('Missing dir', d); ok=False
print('OK' if ok else 'FAIL')
exit(0 if ok else 1)
PY

clean:
    rm -rf .venv __pycache__ */.ipynb_checkpoints

run-smoke:
    python tests/smoke.py

clean-outputs:
    python scripts/clean_outputs.py

run-agent:
    # Run the example agent workflow (uses stub LLM by default)
    ./scripts/run_agent_workflow.sh
