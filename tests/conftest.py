import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a small deterministic sample image file and return its path."""
    p = tmp_path / "sample.jpg"
    w, h = 64, 64
    img = Image.new("RGB", (w, h), color=(10, 20, 30))
    draw = ImageDraw.Draw(img)
    for x in range(w):
        for y in range(h):
            r = int(40 + (x / w) * 200)
            g = int(30 + (y / h) * 180)
            b = 100
            draw.point((x, y), fill=(r, g, b))
    img.save(p)
    return str(p)
