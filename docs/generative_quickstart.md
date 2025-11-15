# Generative Quickstart

This quickstart shows the small, local-first generative tools included in the repo.

Text generation

- `generative_models/gen_utils.py` — small wrapper that uses OpenAI when `OPENAI_API_KEY` is set, otherwise uses a deterministic stub. Use `generate_text(prompt)` to get text.
- `generative_models/batch_text_generate.py` — reads `prompts/generative_prompts.txt` and writes generated text to `./outputs/`.

Multimodal demo

- `multimodal/multimodal_demo.py --image path/to/img.jpg` — computes an image embedding (CLIP if available, otherwise a deterministic fallback), builds a short prompt, and produces a caption (stub/OpenAI as available). Caption is saved to `./outputs/<image>_caption.txt`.

Examples

```bash
# Batch text generation (stub LLM by default)
python generative_models/batch_text_generate.py

# Multimodal caption (safe offline)
python scripts/create_sample_image.py  # creates examples/sample.jpg if you don't have one
python multimodal/multimodal_demo.py --image examples/sample.jpg
```

Notes

- These tools are intentionally conservative: they fall back to deterministic, dependency-light behavior to be runnable in CI and by learners without API keys or GPUs.
- Replace the stub by setting `OPENAI_API_KEY` or by adapting `gen_utils.openai_generate` where appropriate.
