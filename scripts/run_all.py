#!/usr/bin/env python3
import os, json

ROOT = os.path.dirname(os.path.dirname(__file__))
nb_list = []
for base in ("agents","rag_systems","multimodal","generative_models"):
    d = os.path.join(ROOT, base)
    if not os.path.isdir(d):
        continue
    for r,_,files in os.walk(d):
        for f in files:
            if f.endswith('.ipynb'):
                nb_list.append(os.path.join(r,f))

print("Notebooks:")
for p in nb_list:
    print(" -", os.path.relpath(p, ROOT))

out_dir = os.path.join(ROOT, 'outputs')
print("\nOutputs dir:", out_dir, "exists=", os.path.isdir(out_dir))
