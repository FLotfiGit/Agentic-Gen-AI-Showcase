# Diffusion Playground Quickstart

This file explains how to use `generative_models/diffusion_playground.py` to experiment with Stable Diffusion.

Requirements
- Python 3.10+
- `pip install -r requirements.txt`
- (Optional) `HF_TOKEN` environment variable if model requires authentication.

Run the script

```bash
# create venv and install deps
make setup
source .venv/bin/activate
pip install -r requirements.txt

# run the playground (may download model weights)
python generative_models/diffusion_playground.py
```

Tips
- Set `HF_TOKEN` if using private HF models.
- Change `prompt`, `seed`, `steps`, and `guidance` in the script to explore variation.
- If you lack a GPU, set a smaller model or run with fewer steps to reduce runtime.

CPU demo and batch generation

This repository includes a lightweight CPU-friendly demo and a batch runner for learners without GPUs:

- `generative_models/diffusion_cpu_demo.py` — attempts to load a Stable Diffusion pipeline; if loading fails, it falls back to saving a placeholder image with the prompt text (fast, useful for testing).
- `generative_models/prompts.txt` — a small list of example prompts.
- `generative_models/batch_generate.py` — runs the prompts in `prompts.txt` through the CPU demo and saves outputs to `./outputs/`.

Run the batch generator:

```bash
python generative_models/batch_generate.py
```

This is intended for experimentation and learning; replace the prompts or connect to a real model when available.
