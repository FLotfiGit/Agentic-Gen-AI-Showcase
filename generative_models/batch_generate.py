"""Batch generator that reads prompts from prompts.txt and uses diffusion_cpu_demo.generate

Usage:
  python generative_models/batch_generate.py
"""
import os
from generative_models.diffusion_cpu_demo import generate

PROMPTS_FILE = os.path.join(os.path.dirname(__file__), 'prompts.txt')

if __name__ == '__main__':
    if not os.path.exists(PROMPTS_FILE):
        print('No prompts file found at', PROMPTS_FILE)
        raise SystemExit(1)

    with open(PROMPTS_FILE, 'r') as f:
        prompts = [l.strip() for l in f.readlines() if l.strip()]

    for i, p in enumerate(prompts):
        print(f'[{i+1}/{len(prompts)}] Generating for prompt: {p}')
        out = generate(p, seed=100 + i, out_prefix=f'batch_{i+1}')
        print(' ->', out)
