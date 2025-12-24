from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import time


class OutcomeType(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


@dataclass
class Outcome:
    outcome_type: OutcomeType
    result: Any
    feedback: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Iteration:
    iteration_num: int
    input_state: Dict[str, Any]
    output_state: Dict[str, Any]
    outcome: Outcome
    improvements: List[str] = field(default_factory=list)
    duration: float = 0.0


class FeedbackLoop:
    """Iterative refinement system that learns from outcomes."""

    def __init__(self, max_iterations: int = 5, success_threshold: float = 0.8):
        self.max_iterations = max_iterations
        self.success_threshold = success_threshold
        self.iterations: List[Iteration] = []
        self.learned_patterns: Dict[str, Any] = {}

    def run(
        self,
        initial_state: Dict[str, Any],
        action_fn: Callable[[Dict[str, Any]], Any],
        evaluate_fn: Callable[[Any], Outcome],
        improve_fn: Optional[Callable[[Dict[str, Any], Outcome], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run feedback loop until success or max iterations."""
        current_state = initial_state.copy()
        iteration_num = 0

        while iteration_num < self.max_iterations:
            start_time = time.time()
            iteration_num += 1

            # Execute action
            result = action_fn(current_state)

            # Evaluate outcome
            outcome = evaluate_fn(result)

            # Record iteration
            iteration = Iteration(
                iteration_num=iteration_num,
                input_state=current_state.copy(),
                output_state={"result": result},
                outcome=outcome,
                duration=time.time() - start_time,
            )

            # Check for success
            if outcome.outcome_type == OutcomeType.SUCCESS:
                iteration.improvements.append("Goal achieved")
                self.iterations.append(iteration)
                self._learn_from_success(current_state, outcome)
                return {
                    "success": True,
                    "iterations": iteration_num,
                    "final_result": result,
                    "outcome": outcome,
                }

            # Apply improvements if available
            if improve_fn:
                improvements = improve_fn(current_state, outcome)
                current_state.update(improvements)
                iteration.improvements.extend(improvements.keys())
                self._learn_from_failure(current_state, outcome)

            self.iterations.append(iteration)

            # Early stop if no improvement function
            if not improve_fn:
                break

        # Max iterations reached
        return {
            "success": False,
            "iterations": iteration_num,
            "final_result": result if iteration_num > 0 else None,
            "reason": "max_iterations_reached",
        }

    def _learn_from_success(self, state: Dict[str, Any], outcome: Outcome) -> None:
        """Extract patterns from successful outcomes."""
        pattern_key = f"success_{len(self.learned_patterns)}"
        self.learned_patterns[pattern_key] = {
            "state_snapshot": state.copy(),
            "outcome_feedback": outcome.feedback,
            "timestamp": outcome.timestamp,
        }

    def _learn_from_failure(self, state: Dict[str, Any], outcome: Outcome) -> None:
        """Record failure patterns to avoid."""
        pattern_key = f"failure_{len(self.learned_patterns)}"
        self.learned_patterns[pattern_key] = {
            "state_snapshot": state.copy(),
            "outcome_feedback": outcome.feedback,
            "timestamp": outcome.timestamp,
        }

    def get_iteration_summary(self) -> str:
        """Generate summary of all iterations."""
        lines = ["=== Feedback Loop Summary ===\n"]
        for it in self.iterations:
            outcome_icon = {
                OutcomeType.SUCCESS: "✓",
                OutcomeType.PARTIAL: "◐",
                OutcomeType.FAILURE: "✗",
            }
            icon = outcome_icon.get(it.outcome.outcome_type, "?")
            lines.append(
                f"Iteration {it.iteration_num}: {icon} {it.outcome.outcome_type.value} "
                f"({it.duration:.2f}s)"
            )
            lines.append(f"  Feedback: {it.outcome.feedback}")
            if it.improvements:
                lines.append(f"  Improvements: {', '.join(it.improvements)}")

        if self.learned_patterns:
            lines.append(f"\nLearned {len(self.learned_patterns)} patterns")

        return "\n".join(lines)

    def get_convergence_rate(self) -> float:
        """Calculate how quickly the loop converges."""
        if not self.iterations:
            return 0.0

        successful = sum(
            1 for it in self.iterations if it.outcome.outcome_type == OutcomeType.SUCCESS
        )
        return successful / len(self.iterations)

    def reset(self) -> None:
        """Reset feedback loop state."""
        self.iterations.clear()
        self.learned_patterns.clear()


class AdaptiveFeedback:
    """Adaptive feedback that adjusts strategy based on history."""

    def __init__(self):
        self.strategy_scores: Dict[str, List[float]] = {}
        self.current_strategy = "default"

    def record_strategy_outcome(self, strategy: str, score: float) -> None:
        """Record outcome score for a strategy."""
        if strategy not in self.strategy_scores:
            self.strategy_scores[strategy] = []
        self.strategy_scores[strategy].append(score)

    def select_best_strategy(self) -> str:
        """Select strategy with highest average score."""
        if not self.strategy_scores:
            return self.current_strategy

        avg_scores = {
            strategy: sum(scores) / len(scores)
            for strategy, scores in self.strategy_scores.items()
        }

        best = max(avg_scores.items(), key=lambda x: x[1])
        self.current_strategy = best[0]
        return best[0]

    def get_strategy_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all strategies."""
        stats = {}
        for strategy, scores in self.strategy_scores.items():
            stats[strategy] = {
                "avg": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "count": len(scores),
            }
        return stats


class ReinforcementSignal:
    """Simple reinforcement signal for action selection."""

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.action_values: Dict[str, float] = {}

    def update(self, action: str, reward: float) -> None:
        """Update action value using simple moving average."""
        if action not in self.action_values:
            self.action_values[action] = 0.0

        old_value = self.action_values[action]
        self.action_values[action] = old_value + self.learning_rate * (reward - old_value)

    def select_action(self, available_actions: List[str], epsilon: float = 0.1) -> str:
        """Select action using epsilon-greedy strategy."""
        import random

        # Exploration
        if random.random() < epsilon:
            return random.choice(available_actions)

        # Exploitation - choose best known action
        best_action = max(
            available_actions,
            key=lambda a: self.action_values.get(a, 0.0),
        )
        return best_action

    def get_action_rankings(self) -> List[tuple]:
        """Get actions ranked by value."""
        return sorted(self.action_values.items(), key=lambda x: x[1], reverse=True)
