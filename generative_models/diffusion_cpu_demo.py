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

OUTPUTS = Path("./outputs")
OUTPUTS.mkdir(exist_ok=True)


def _fallback_image(prompt: str, out_prefix: str = "cpu_demo", seed: int = 0):
    """Create a simple placeholder image with the prompt text for quick experiments."""
    w, h = 768, 512
    img = Image.new("RGB", (w, h), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    text = f"PROMPT:\n{prompt}"[:1000]
    draw.multiline_text((20, 20), text, fill=(230, 230, 230), font=font)
    ts = int(time.time())
    out_path = OUTPUTS / f"{out_prefix}_{ts}.png"
    img.save(out_path)
    return str(out_path)


def try_load_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5"):
    """Try to import and load a pipeline; return None on failure (informative)."""
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except Exception as e:
        return None, f"Missing diffusers/torch: {e}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = os.getenv("HF_TOKEN")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else None, use_auth_token=hf_token)
        pipe = pipe.to(device)
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
        generator = torch.Generator(device=pipe.device).manual_seed(seed) if torch.cuda.is_available() else None
    except Exception:
        generator = None

    res = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, generator=generator)
    img = res.images[0]
    ts = int(time.time())
    out_path = OUTPUTS / f"{out_prefix}_{ts}.png"
    img.save(out_path)
    print("Saved generated image to", out_path)
    return str(out_path)


if __name__ == "__main__":
    test_prompt = os.getenv("DEMO_PROMPT", "A small red robot reading a book, cozy scene")
    generate(test_prompt)
