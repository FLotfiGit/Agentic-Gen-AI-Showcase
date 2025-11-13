from generative_models import gen_utils


def test_stub_generation_basic():
    out = gen_utils.generate_text("Please summarize this text: Hello world", max_tokens=10)
    assert isinstance(out, str)
    assert len(out) > 0


def test_caption_prompt_from_embedding():
    emb = [0.1] * 16
    p = gen_utils.caption_prompt_from_image_embedding(emb)
    assert "Embedding-norm" in p
