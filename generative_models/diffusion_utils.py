"""Small utilities for diffusion scripts.

Helpers: ensure output dir, timestamped filenames, image saving helpers and device detection.
"""
from pathlib import Path
import time
from PIL import Image


def ensure_output_dir(path: str = "./outputs") -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamped_filename(prefix: str = "image", prompt: str = "", ext: str = ".png") -> str:
    ts = int(time.time())
    safe_prompt = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in prompt)[:40].strip().replace(" ", "_")
    name = f"{prefix}_{safe_prompt}_{ts}{ext}" if prompt else f"{prefix}_{ts}{ext}"
    return name


def save_pil_image(img: Image.Image, out_dir: str = "./outputs", name: str | None = None, prefix: str = "image") -> str:
    out = ensure_output_dir(out_dir)
    if name is None:
        name = timestamped_filename(prefix=prefix)
    path = out / name
    img.save(path)
    return str(path)


def get_default_device():
    """Return a string device hint: 'cuda', 'mps', or 'cpu'."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
