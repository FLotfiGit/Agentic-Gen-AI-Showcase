"""Multimodal helpers package.

Provides lightweight helpers for image loading/processing used across demos.
"""
from .utils import load_image, image_to_tensor_stub, get_image_embedding  # noqa: F401

__all__ = ["load_image", "image_to_tensor_stub", "get_image_embedding"]

__version__ = "0.1.0"
