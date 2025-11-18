"""Retrieval utilities package.

Expose the Retriever and faiss helpers for easy imports in tests and demos.
"""
from .retriever import Retriever  # noqa: F401
from .faiss_index import build_embeddings, build_faiss_index  # noqa: F401

__all__ = ["Retriever", "build_embeddings", "build_faiss_index"]

__version__ = "0.1.0"
