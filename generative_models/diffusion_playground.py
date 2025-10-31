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


def get_default_device():
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        # Apple Silicon MPS
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_pipeline(model_id="runwayml/stable-diffusion-v1-5", device=None, hf_token_env="HF_TOKEN"):
    """Load a Stable Diffusion pipeline with safe defaults.

    Returns pipeline or raises informative error.
    """
    if device is None:
        device = get_default_device()

    hf_token = os.getenv(hf_token_env)
    auth = None if hf_token is None else hf_token

    print(f"Loading model {model_id} on {device} (HF token set={hf_token is not None})")
    # Choose dtype conservatively
    torch_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=auth, torch_dtype=torch_dtype)
    try:
        pipe = pipe.to(device)
    except Exception:
        # Some pipelines may not support direct .to(device) for mps; rely on inference to move tensors
        print("Warning: could not call .to(device) on pipeline; continuing and hoping inference handles device placement")
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

    # Build a generator where possible for reproducibility
    generator = None
    try:
        dev = get_default_device()
        if dev == "cuda":
            generator = torch.Generator(device="cuda").manual_seed(seed)
        elif dev == "mps":
            generator = torch.Generator(device="cpu").manual_seed(seed)
        else:
            generator = torch.Generator().manual_seed(seed)
    except Exception:
        generator = None

    result = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, generator=generator)
    image = result.images[0]

    ts = int(time.time())
    safe_prompt = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in prompt)[:40].strip().replace(" ", "_")
    out_path = OUTPUTS / f"{out_prefix}_{safe_prompt}_{ts}.png"
    image.save(out_path)
    print("Saved image to", out_path)
    return str(out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a simple stable diffusion generation (guarded)")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="HF model id to load")
    parser.add_argument("--prompt", default="A futuristic AI agent in a neon-lit digital city, cinematic lighting")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=8.0)
    parser.add_argument("--device", default=None, help="Override device: cpu, cuda, or mps")
    args = parser.parse_args()

    try:
        pipe = load_pipeline(model_id=args.model, device=args.device)
    except Exception as e:
        print("Failed to load pipeline:", e)
        print("If you don't have a GPU or model access, consider setting HF_TOKEN or using the CPU demo script.")
        raise

    out = generate_image(pipe, prompt=args.prompt, seed=args.seed, steps=args.steps, guidance=args.guidance)
    print(f"Open {out} to view the generated image. You can change seed/guidance via CLI args.")
