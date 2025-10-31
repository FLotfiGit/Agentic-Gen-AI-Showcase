import os
from pathlib import Path
from generative_models import diffusion_utils
from PIL import Image


def test_timestamped_filename_contains_prefix_and_ext():
    name = diffusion_utils.timestamped_filename(prefix="testprefix", prompt="a dog")
    assert name.startswith("testprefix_")
    assert name.endswith(".png")


def test_ensure_output_dir_and_save(tmp_path):
    outdir = tmp_path / "outs"
    p = diffusion_utils.ensure_output_dir(str(outdir))
    assert p.exists()
    # Save a small image
    img = Image.new("RGB", (8, 8), color=(10, 20, 30))
    saved = diffusion_utils.save_pil_image(img, out_dir=str(outdir), name="small.png")
    assert Path(saved).exists()
