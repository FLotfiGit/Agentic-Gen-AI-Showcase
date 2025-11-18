## Multimodal quickstart

This short quickstart shows how to produce a deterministic sample image and run the multimodal demo locally.

1. Create or generate the sample image:

```bash
python scripts/create_sample_image.py --output examples/sample.jpg
```

2. Run the multimodal demo (uses deterministic fallback if CLIP/transformers aren't installed):

```bash
PYTHONPATH=. python multimodal/multimodal_demo.py --image examples/sample.jpg
```

3. Run the lightweight tests locally:

```bash
PYTHONPATH=. pytest -q tests/test_multimodal.py
```

Notes:
- The repository provides `tests/conftest.py` to create a deterministic temporary image used by tests.
- The CI workflow `.github/workflows/light-tests.yml` runs the same lightweight multimodal test to ensure basic functionality on PRs.
