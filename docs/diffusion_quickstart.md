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
