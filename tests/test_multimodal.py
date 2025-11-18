import os
from multimodal.utils import load_image, image_to_tensor_stub, get_image_embedding


def test_load_image_and_tensor(sample_image_path):
    img = load_image(sample_image_path)
    assert img is not None
    vec = image_to_tensor_stub(img, size=(32, 32))
    # flattened tensor size should match 32*32*3
    assert len(vec) == 32 * 32 * 3


def test_get_image_embedding_deterministic(sample_image_path):
    emb1 = get_image_embedding(sample_image_path, dim=128)
    emb2 = get_image_embedding(sample_image_path, dim=128)
    # embeddings should be deterministic and have the requested dimension
    assert len(emb1) == 128
    assert len(emb2) == 128
    assert emb1 == emb2
