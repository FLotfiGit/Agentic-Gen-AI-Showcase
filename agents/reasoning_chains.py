from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from enum import Enum


class ReasoningStep(Enum):
    OBSERVE = "observe"
    THINK = "think"
    ACT = "act"
    REFLECT = "reflect"


@dataclass
class ReasoningTrace:
    step_type: ReasoningStep
    content: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ChainOfThought:
    """Chain-of-thought reasoning with step-by-step breakdown."""

    def __init__(self):
        self.traces: List[ReasoningTrace] = []

    def observe(self, observation: str, confidence: float = 1.0) -> None:
        """Record an observation."""
        self.traces.append(
            ReasoningTrace(
                step_type=ReasoningStep.OBSERVE,
                content=observation,
                confidence=confidence,
            )
        )

    def think(self, thought: str, confidence: float = 1.0) -> None:
        """Record a reasoning step."""
        self.traces.append(
            ReasoningTrace(
                step_type=ReasoningStep.THINK,
                content=thought,
                confidence=confidence,
            )
        )

    def act(self, action: str, confidence: float = 1.0) -> None:
        """Record an action decision."""
        self.traces.append(
            ReasoningTrace(
                step_type=ReasoningStep.ACT,
                content=action,
                confidence=confidence,
            )
        )

    def get_trace(self) -> List[ReasoningTrace]:
        """Get the full reasoning trace."""
        return self.traces.copy()

    def format_trace(self) -> str:
        """Format trace as readable text."""
        lines = ["=== Chain of Thought ===\n"]
        for i, trace in enumerate(self.traces, 1):
            lines.append(f"{i}. [{trace.step_type.value.upper()}] {trace.content}")
            if trace.confidence < 1.0:
                lines.append(f"   (confidence: {trace.confidence:.2f})")
        return "\n".join(lines)


class SelfReflection:
    """Self-reflection mechanism for evaluating reasoning quality."""

    def __init__(self, cot: ChainOfThought):
        self.cot = cot
        self.reflections: List[ReasoningTrace] = []

    def reflect(self, reflection: str, confidence: float = 1.0) -> None:
        """Add a reflection on the reasoning process."""
        self.reflections.append(
            ReasoningTrace(
                step_type=ReasoningStep.REFLECT,
                content=reflection,
                confidence=confidence,
            )
        )

    def evaluate_confidence(self) -> float:
        """Compute average confidence across reasoning steps."""
        if not self.cot.traces:
            return 0.0
        return sum(t.confidence for t in self.cot.traces) / len(self.cot.traces)

    def identify_weak_steps(self, threshold: float = 0.7) -> List[int]:
        """Find reasoning steps with low confidence."""
        return [
            i for i, trace in enumerate(self.cot.traces)
            if trace.confidence < threshold
        ]

    def format_reflection(self) -> str:
        """Format reflections as readable text."""
        lines = ["\n=== Self-Reflection ===\n"]
        lines.append(f"Average confidence: {self.evaluate_confidence():.2f}")
        weak = self.identify_weak_steps()
        if weak:
            lines.append(f"Weak steps: {weak}")
        lines.append("\nReflections:")
        for i, ref in enumerate(self.reflections, 1):
            lines.append(f"{i}. {ref.content}")
        return "\n".join(lines)


class ReasoningChain:
    """Composable reasoning chain with validation."""

    def __init__(self):
        self.steps: List[Callable[[Any], Any]] = []
        self.validators: List[Callable[[Any], bool]] = []

    def add_step(
        self,
        step: Callable[[Any], Any],
        validator: Optional[Callable[[Any], bool]] = None,
    ) -> ReasoningChain:
        """Add a reasoning step with optional validation."""
        self.steps.append(step)
        self.validators.append(validator or (lambda x: True))
        return self

    def execute(self, initial_input: Any) -> Dict[str, Any]:
        """Execute the reasoning chain."""
        current = initial_input
        trace = []

        for i, (step, validator) in enumerate(zip(self.steps, self.validators)):
            try:
                result = step(current)
                is_valid = validator(result)
                trace.append({
                    "step": i,
                    "input": str(current)[:100],
                    "output": str(result)[:100],
                    "valid": is_valid,
                })
                if not is_valid:
                    return {
                        "success": False,
                        "trace": trace,
                        "error": f"Validation failed at step {i}",
                    }
                current = result
            except Exception as e:
                trace.append({
                    "step": i,
                    "error": str(e),
                })
                return {
                    "success": False,
                    "trace": trace,
                    "error": str(e),
                }

        return {
            "success": True,
            "result": current,
            "trace": trace,
        }


class TreeOfThoughts:
    """Tree-of-thoughts with branching and backtracking."""

    @dataclass
    class Node:
        content: str
        score: float = 0.0
        children: List[TreeOfThoughts.Node] = None
        parent: Optional[TreeOfThoughts.Node] = None

        def __post_init__(self):
            if self.children is None:
                self.children = []

    def __init__(self, root_content: str):
        self.root = self.Node(content=root_content)
        self.current = self.root

    def branch(self, thoughts: List[str], scores: Optional[List[float]] = None) -> None:
        """Create branches from current node."""
        if scores is None:
            scores = [1.0] * len(thoughts)

        for thought, score in zip(thoughts, scores):
            node = self.Node(content=thought, score=score, parent=self.current)
            self.current.children.append(node)

    def select_best_child(self) -> Optional[Node]:
        """Select highest-scoring child."""
        if not self.current.children:
            return None
        return max(self.current.children, key=lambda n: n.score)

    def navigate_to(self, node: Node) -> None:
        """Navigate to a specific node."""
        self.current = node

    def get_path(self) -> List[str]:
        """Get path from root to current node."""
        path = []
        node = self.current
        while node:
            path.append(node.content)
            node = node.parent
        return list(reversed(path))

    def format_tree(self, node: Optional[Node] = None, depth: int = 0) -> str:
        """Format tree as readable text."""
        if node is None:
            node = self.root

        lines = []
        indent = "  " * depth
        marker = "→ " if node == self.current else ""
        lines.append(f"{indent}{marker}{node.content} (score: {node.score:.2f})")

        for child in node.children:
            lines.append(self.format_tree(child, depth + 1))

        return "\n".join(lines)
