"""Diffusion Playground

A small starter script to experiment with text-to-image diffusion models (Stable Diffusion).

Usage:
  - Create and activate a Python 3.10+ virtualenv (see README)
  - Install dependencies: pip install -r requirements.txt
  - Optionally set HF_TOKEN in environment if required by model
  - Run: python generative_models/diffusion_playground.py

The script is intentionally defensive: it checks for GPU and for missing tokens and falls back or prints helpful messages.
"""

import os
import time
from pathlib import Path

# Lightweight runtime guards
try:
    import torch
    from diffusers import StableDiffusionPipeline
except Exception as e:
    print("Missing package for diffusion:", e)
    print("Install dependencies with: pip install -r requirements.txt")
    raise

OUTPUTS = Path("./outputs")
OUTPUTS.mkdir(parents=True, exist_ok=True)


def load_pipeline(model_id="runwayml/stable-diffusion-v1-5", device=None, hf_token_env="HF_TOKEN"):
    """Load a Stable Diffusion pipeline with safe defaults.

    Returns pipeline or raises informative error.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_token = os.getenv(hf_token_env)
    auth = None if hf_token is None else hf_token

    print(f"Loading model {model_id} on {device} (HF token set={hf_token is not None})")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=auth, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32)
    pipe = pipe.to(device)
    return pipe


def generate_image(pipe, prompt="A photorealistic painting of an AI agent", seed=42, steps=30, guidance=7.5, out_prefix="diffusion_demo"):
    """Generate an image with the pipeline and save to ./outputs/ with timestamp.

    Returns output path.
    """
    if hasattr(pipe, "enable_attention_slicing"):
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    generator = torch.Generator(device=pipe.device).manual_seed(seed) if torch.cuda.is_available() or torch.backends.mps.is_available() else None
    result = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, generator=generator)
    image = result.images[0]

    ts = int(time.time())
    out_path = OUTPUTS / f"{out_prefix}_{ts}.png"
    image.save(out_path)
    print("Saved image to", out_path)
    return str(out_path)


if __name__ == "__main__":
    try:
        pipe = load_pipeline()
    except Exception as e:
        print("Failed to load pipeline:", e)
        print("If you don't have a GPU or model access, consider setting HF_TOKEN or using a small local model.")
        raise

    # Example generation
    prompt = "A futuristic AI agent in a neon-lit digital city, cinematic lighting"
    generate_image(pipe, prompt=prompt, seed=1234, steps=25, guidance=8.0)
