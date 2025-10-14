#!/usr/bin/env python3
import os, importlib.util, sys

REQUIRED_DIRS = ["agents","rag_systems","multimodal","generative_models","outputs"]
missing = [d for d in REQUIRED_DIRS if not os.path.isdir(d)]
if missing:
    print("Missing directories:", missing)
    sys.exit(1)
print("All required directories present.")

# Basic import smoke for evaluator stub if exists
if os.path.exists("evaluation/evaluator.py"):
    spec = importlib.util.spec_from_file_location("evaluator","evaluation/evaluator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, 'Evaluator'):
        print('Evaluator class found.')

print("Smoke test passed.")
