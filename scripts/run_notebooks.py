#!/usr/bin/env python3
"""Execute all example notebooks (guards for long runs)."""
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = os.path.dirname(os.path.dirname(__file__))
notebooks = []
for base in ("agents","rag_systems","multimodal","generative_models"):
    d = os.path.join(ROOT, base)
    if not os.path.isdir(d):
        continue
    for r,_,files in os.walk(d):
        for f in files:
            if f.endswith('.ipynb'):
                notebooks.append(os.path.join(r,f))

os.makedirs(os.path.join(ROOT, 'outputs'), exist_ok=True)

for nb_path in notebooks:
    print('Executing', nb_path)
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(nb_path)}})
        out_path = os.path.join(ROOT, 'outputs', 'executed_' + os.path.basename(nb_path))
        with open(out_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print('Saved executed notebook to', out_path)
    except Exception as e:
        print('Execution failed for', nb_path, 'error:', e)
        raise
