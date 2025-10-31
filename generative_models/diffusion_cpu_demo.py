"""CPU-friendly Diffusion Demo

This script attempts to load a Stable Diffusion pipeline. If no suitable device or model
access is available, it falls back to a small "fake" generator that produces a simple
placeholder image and saves it to ./outputs/.

Intended for learners who don't have GPU access or HF tokens.
"""
import os
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from generative_models.diffusion_utils import ensure_output_dir, save_pil_image, timestamped_filename, get_default_device

OUTPUTS = ensure_output_dir("./outputs")


def _fallback_image(prompt: str, out_prefix: str = "cpu_demo", seed: int = 0):
    """Create a nicer placeholder image with a colored gradient and the prompt text.

    Uses only PIL and numpy so it runs on CPU quickly.
    """
    import numpy as np

    w, h = 768, 512
    # Create a smooth horizontal gradient influenced by seed
    rng = np.random.RandomState(seed)
    base = rng.rand(3)
    x = np.linspace(0, 1, w)
    grad = np.outer(np.ones(h), x)
    img_arr = (np.clip((base * grad[..., None]) * 255, 0, 255)).astype("uint8")
    # Add a soft vignette
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    vignette = 1 - (dist / dist.max()) * 0.6
    img_arr = (img_arr * vignette[..., None]).astype("uint8")
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    text = f"PROMPT: {prompt}"[:800]
    draw.text((16, 16), text, fill=(255, 255, 255), font=font)
    name = timestamped_filename(prefix=out_prefix, prompt=prompt)
    path = save_pil_image(img, out_dir=str(OUTPUTS), name=name, prefix=out_prefix)
    return path


def try_load_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5"):
    """Try to import and load a pipeline; return None on failure (informative)."""
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except Exception as e:
        return None, f"Missing diffusers/torch: {e}"

    device = get_default_device()
    hf_token = os.getenv("HF_TOKEN")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else None, use_auth_token=hf_token)
        try:
            pipe = pipe.to(device)
        except Exception:
            print("Warning: could not move pipeline to device; continuing")
        return pipe, None
    except Exception as e:
        return None, f"Failed to load model {model_id}: {e}"


def generate(prompt: str, seed: int = 42, steps: int = 20, guidance: float = 7.5, out_prefix: str = "cpu_demo"):
    """Generate an image using a loaded pipeline or fall back to the placeholder generator."""
    pipe, err = try_load_pipeline()
    if pipe is None:
        print("Falling back to placeholder image generation because:", err)
        path = _fallback_image(prompt, out_prefix=out_prefix, seed=seed)
        print("Saved fallback image to", path)
        return path

    # If pipeline loaded, generate (beware: may be slow on CPU)
    print("Pipeline loaded; generating (may be slow on CPU)…")
    if hasattr(pipe, "enable_attention_slicing"):
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    generator = None
    try:
        import torch
        dev = get_default_device()
        if dev == "cuda":
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = torch.Generator().manual_seed(seed)
    except Exception:
        generator = None

    res = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, generator=generator)
    img = res.images[0]
    name = timestamped_filename(prefix=out_prefix, prompt=prompt)
    path = save_pil_image(img, out_dir=str(OUTPUTS), name=name, prefix=out_prefix)
    print("Saved generated image to", path)
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CPU-friendly demo for diffusion learners")
    parser.add_argument("--prompt", default=os.getenv("DEMO_PROMPT", "A small red robot reading a book, cozy scene"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    args = parser.parse_args()

    generate(args.prompt, seed=args.seed, steps=args.steps, guidance=args.guidance, out_prefix="cpu_demo")
