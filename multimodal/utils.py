"""Multimodal helpers: image preprocessing and optional CLIP embedding wrapper.

If CLIP (transformers + CLIPModel) is not installed, the embedding function returns a fixed-size random vector
seeded by the image bytes so it's deterministic for the same image (useful for demos/tests).
"""
from pathlib import Path
from typing import List
from PIL import Image
import numpy as np
import hashlib


def load_image(path: str) -> Image.Image:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    img = Image.open(p).convert("RGB")
    return img


def image_to_tensor_stub(img: Image.Image, size=(224, 224)):
    # Resize and normalize to [0,1]
    img = img.resize(size)
    arr = np.array(img).astype("float32") / 255.0
    # return flattened vector for simplicity
    return arr.flatten()


def get_image_embedding(path: str, model_name: str = None, dim: int = 512):
    """Return an embedding vector for an image.

    If CLIP is available, use it. Otherwise return a deterministic pseudo-embedding based on image bytes.
    """
    # Try to use CLIP via transformers if installed
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        model_name = model_name or "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        image = load_image(path)
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
        emb = emb[0].cpu().numpy()
        return emb.tolist()
    except Exception:
        # fallback deterministic random vector seeded by image bytes
        b = Path(path).read_bytes()
        h = hashlib.sha256(b).digest()
        rng = np.random.RandomState(int.from_bytes(h[:8], "big"))
        vec = rng.randn(dim).astype("float32")
        # normalize
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec.tolist()
