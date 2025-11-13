"""Simple multimodal demo: load an image, compute embedding, and ask LLM for a caption via `gen_utils`.

This demo is safe to run offline: if CLIP/OpenAI are missing it falls back to deterministic stubs.
"""
import argparse
from multimodal.utils import load_image, get_image_embedding
from generative_models.gen_utils import caption_prompt_from_image_embedding, generate_text
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an input image")
    parser.add_argument("--out", default="./outputs", help="Output directory to save caption.txt")
    args = parser.parse_args()

    p = Path(args.image)
    if not p.exists():
        raise FileNotFoundError(args.image)

    emb = get_image_embedding(str(p))
    prompt = caption_prompt_from_image_embedding(emb)
    caption = generate_text(prompt, max_tokens=50)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    out_file = outdir / (p.stem + "_caption.txt")
    out_file.write_text(caption, encoding="utf-8")
    print("Saved caption to", out_file)


if __name__ == "__main__":
    main()
