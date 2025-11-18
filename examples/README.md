# Examples

This folder includes guidance for example assets used by demos and tests.

sample.jpg
-----------

We intentionally do not commit large binary images to the repository. To run
multimodal demos and tests that expect an example image, either:

- Run the sample image generator script:

```bash
python scripts/create_sample_image.py --output examples/sample.jpg
```

- Or copy a small JPEG image to `examples/sample.jpg`.

The tests/fixtures will generate a temporary deterministic image during tests
if no `examples/sample.jpg` is present.
