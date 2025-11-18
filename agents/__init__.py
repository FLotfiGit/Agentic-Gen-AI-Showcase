"""Agents package exports.

This file makes `agents` importable as a package and provides a small
convenience import for common symbols used in tests and demos.
"""
from .agent_utils import SimpleAgent, stub_llm, Thought, Action  # noqa: F401
from .planner import decompose_goal  # noqa: F401

__all__ = ["SimpleAgent", "stub_llm", "Thought", "Action", "decompose_goal"]

__version__ = "0.1.0"
