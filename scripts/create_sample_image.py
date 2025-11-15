"""Create a small deterministic sample image for demos/tests.

This script writes `examples/sample.jpg` (RGB 128x128) with a simple gradient and text.
Run: python scripts/create_sample_image.py
"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def create_sample(path="examples/sample.jpg"):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    w, h = 128, 128
    img = Image.new("RGB", (w, h), color=(30, 30, 60))
    draw = ImageDraw.Draw(img)
    # simple gradient
    for x in range(w):
        for y in range(h):
            r = int(50 + (x / w) * 205)
            g = int(50 + (y / h) * 155)
            b = 120
            draw.point((x, y), fill=(r, g, b))
    try:
        font = ImageFont.load_default()
        draw.text((8, h - 20), "sample", font=font, fill=(255, 255, 255))
    except Exception:
        pass
    img.save(p)
    print("Wrote sample image to", p)


if __name__ == "__main__":
    create_sample()
