"""Generative models helpers package.

Exports commonly used generative utilities for easier imports.
"""
from .gen_utils import generate_text, stub_generate  # noqa: F401
from .diffusion_utils import safe_diffusion_seed  # noqa: F401

__all__ = ["generate_text", "stub_generate", "safe_diffusion_seed"]

__version__ = "0.1.0"
"""Generative models package exports for convenience.

Keep this light — it simply exposes the main diffusion helpers.
"""
from . import diffusion_utils

__all__ = ["diffusion_utils"]
