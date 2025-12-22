from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class CritiqueType(Enum):
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    EFFICIENCY = "efficiency"
    CLARITY = "clarity"
    SAFETY = "safety"


@dataclass
class Critique:
    critique_type: CritiqueType
    score: float  # 0.0 to 1.0
    feedback: str
    suggestions: List[str]

    def is_passing(self, threshold: float = 0.7) -> bool:
        """Check if critique score meets threshold."""
        return self.score >= threshold


class AgentCritic:
    """Self-critique system for agent outputs and reasoning."""

    def __init__(self, thresholds: Optional[Dict[CritiqueType, float]] = None):
        self.thresholds = thresholds or {
            CritiqueType.CORRECTNESS: 0.8,
            CritiqueType.COMPLETENESS: 0.7,
            CritiqueType.EFFICIENCY: 0.6,
            CritiqueType.CLARITY: 0.7,
            CritiqueType.SAFETY: 0.9,
        }
        self.critique_history: List[Critique] = []

    def critique_output(
        self,
        output: str,
        expected_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[Critique]:
        """Critique an agent output against multiple dimensions."""
        critiques = []

        # Correctness check
        correctness = self._assess_correctness(output, expected_criteria)
        critiques.append(correctness)

        # Completeness check
        completeness = self._assess_completeness(output, expected_criteria)
        critiques.append(completeness)

        # Clarity check
        clarity = self._assess_clarity(output)
        critiques.append(clarity)

        # Safety check
        safety = self._assess_safety(output)
        critiques.append(safety)

        self.critique_history.extend(critiques)
        return critiques

    def _assess_correctness(
        self,
        output: str,
        criteria: Optional[Dict[str, Any]],
    ) -> Critique:
        """Assess correctness of output."""
        score = 1.0
        feedback = "Output appears correct"
        suggestions = []

        if not output or len(output.strip()) == 0:
            score = 0.0
            feedback = "Empty output"
            suggestions.append("Provide non-empty response")

        if criteria and "expected_format" in criteria:
            expected = criteria["expected_format"]
            if expected == "json" and not self._is_valid_json(output):
                score *= 0.5
                feedback = "Expected JSON format not detected"
                suggestions.append("Format output as valid JSON")

        if criteria and "required_keywords" in criteria:
            keywords = criteria["required_keywords"]
            missing = [kw for kw in keywords if kw.lower() not in output.lower()]
            if missing:
                score *= max(0.3, 1.0 - len(missing) * 0.2)
                feedback = f"Missing required keywords: {missing}"
                suggestions.append(f"Include keywords: {', '.join(missing)}")

        return Critique(
            critique_type=CritiqueType.CORRECTNESS,
            score=score,
            feedback=feedback,
            suggestions=suggestions,
        )

    def _assess_completeness(
        self,
        output: str,
        criteria: Optional[Dict[str, Any]],
    ) -> Critique:
        """Assess completeness of output."""
        score = 0.8
        feedback = "Output appears reasonably complete"
        suggestions = []

        if criteria and "min_length" in criteria:
            min_len = criteria["min_length"]
            if len(output) < min_len:
                score = min(0.8, len(output) / min_len)
                feedback = f"Output too short ({len(output)} < {min_len})"
                suggestions.append(f"Expand output to at least {min_len} characters")

        if criteria and "required_sections" in criteria:
            sections = criteria["required_sections"]
            missing = [s for s in sections if s.lower() not in output.lower()]
            if missing:
                score *= max(0.4, 1.0 - len(missing) * 0.3)
                feedback = f"Missing required sections: {missing}"
                suggestions.append(f"Add sections: {', '.join(missing)}")

        return Critique(
            critique_type=CritiqueType.COMPLETENESS,
            score=score,
            feedback=feedback,
            suggestions=suggestions,
        )

    def _assess_clarity(self, output: str) -> Critique:
        """Assess clarity and readability."""
        score = 0.9
        feedback = "Output is clear"
        suggestions = []

        # Check for extremely long sentences
        sentences = output.split(".")
        long_sentences = [s for s in sentences if len(s) > 200]
        if long_sentences:
            score = max(0.6, 1.0 - len(long_sentences) * 0.1)
            feedback = "Some sentences are too long"
            suggestions.append("Break long sentences into shorter ones")

        # Check for structure
        if len(output) > 500 and "\n" not in output:
            score *= 0.8
            feedback = "Long output lacks structure"
            suggestions.append("Add paragraph breaks for readability")

        return Critique(
            critique_type=CritiqueType.CLARITY,
            score=score,
            feedback=feedback,
            suggestions=suggestions,
        )

    def _assess_safety(self, output: str) -> Critique:
        """Assess safety of output."""
        score = 1.0
        feedback = "Output appears safe"
        suggestions = []

        # Check for potential unsafe patterns (basic heuristics)
        unsafe_patterns = ["delete all", "rm -rf", "drop database", "format c:"]
        detected = [p for p in unsafe_patterns if p in output.lower()]

        if detected:
            score = 0.3
            feedback = f"Detected potentially unsafe commands: {detected}"
            suggestions.append("Review and sanitize unsafe operations")

        return Critique(
            critique_type=CritiqueType.SAFETY,
            score=score,
            feedback=feedback,
            suggestions=suggestions,
        )

    def _is_valid_json(self, text: str) -> bool:
        """Check if text is valid JSON."""
        import json

        try:
            json.loads(text)
            return True
        except:
            return False

    def should_retry(self, critiques: List[Critique]) -> bool:
        """Determine if output should be regenerated based on critiques."""
        for critique in critiques:
            threshold = self.thresholds.get(critique.critique_type, 0.7)
            if not critique.is_passing(threshold):
                return True
        return False

    def get_improvement_prompt(self, critiques: List[Critique]) -> str:
        """Generate improvement suggestions based on critiques."""
        lines = ["Please improve the output based on these critiques:\n"]

        for critique in critiques:
            if not critique.is_passing(self.thresholds.get(critique.critique_type, 0.7)):
                lines.append(f"• {critique.critique_type.value.upper()}: {critique.feedback}")
                for suggestion in critique.suggestions:
                    lines.append(f"  - {suggestion}")

        return "\n".join(lines)

    def format_report(self, critiques: List[Critique]) -> str:
        """Format critique results as a report."""
        lines = ["=== Agent Critique Report ===\n"]

        for critique in critiques:
            passing = "✓" if critique.is_passing(self.thresholds.get(critique.critique_type, 0.7)) else "✗"
            lines.append(f"{passing} {critique.critique_type.value.upper()}: {critique.score:.2f}")
            lines.append(f"   {critique.feedback}")
            if critique.suggestions:
                for suggestion in critique.suggestions:
                    lines.append(f"   → {suggestion}")

        overall = sum(c.score for c in critiques) / len(critiques) if critiques else 0.0
        lines.append(f"\nOverall Score: {overall:.2f}")

        return "\n".join(lines)
