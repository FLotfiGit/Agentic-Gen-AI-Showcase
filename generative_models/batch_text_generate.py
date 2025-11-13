"""Batch text generator: reads prompts from `prompts/generative_prompts.txt` and writes outputs to ./outputs/"""
from generative_models.gen_utils import generate_text
from pathlib import Path


def main():
    p = Path("prompts/generative_prompts.txt")
    if not p.exists():
        print("No prompts file found at prompts/generative_prompts.txt")
        return
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, line in enumerate([l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()], start=1):
        res = generate_text(line, max_tokens=100)
        out_file = out_dir / f"batch_gen_{i}.txt"
        out_file.write_text(res, encoding="utf-8")
        print("Wrote", out_file)


if __name__ == "__main__":
    main()
