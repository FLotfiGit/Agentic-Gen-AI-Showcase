from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time


@dataclass
class ContextWindow:
    content: str
    timestamp: float
    token_estimate: int
    importance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextManager:
    """Manages long-term context with summarization and pruning."""

    def __init__(self, max_tokens: int = 4000, summarization_threshold: float = 0.8):
        self.max_tokens = max_tokens
        self.summarization_threshold = summarization_threshold
        self.windows: List[ContextWindow] = []
        self.summaries: List[str] = []
        self.current_token_count = 0

    def add_context(
        self,
        content: str,
        importance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add new context to the manager."""
        token_est = self._estimate_tokens(content)

        window = ContextWindow(
            content=content,
            timestamp=time.time(),
            token_estimate=token_est,
            importance=importance,
            metadata=metadata or {},
        )

        self.windows.append(window)
        self.current_token_count += token_est

        # Check if we need to compress
        if self.current_token_count > self.max_tokens * self.summarization_threshold:
            self._compress_context()

    def get_context(self, max_tokens: Optional[int] = None) -> str:
        """Retrieve context within token budget."""
        budget = max_tokens or self.max_tokens
        parts = []
        used_tokens = 0

        # Include summaries first
        for summary in self.summaries:
            summary_tokens = self._estimate_tokens(summary)
            if used_tokens + summary_tokens <= budget * 0.3:  # Reserve 30% for summaries
                parts.append(f"[Summary] {summary}")
                used_tokens += summary_tokens

        # Include recent windows in reverse order (most recent first)
        for window in reversed(self.windows):
            if used_tokens + window.token_estimate <= budget:
                parts.insert(len(parts), window.content)
                used_tokens += window.token_estimate
            else:
                break

        return "\n\n".join(parts)

    def _compress_context(self) -> None:
        """Compress old context through summarization and pruning."""
        if len(self.windows) < 3:
            return

        # Find oldest low-importance windows to compress
        sorted_windows = sorted(
            self.windows[: len(self.windows) // 2],  # Only consider first half
            key=lambda w: (w.importance, -w.timestamp),
        )

        # Take bottom 40% for compression
        to_compress = sorted_windows[: max(1, len(sorted_windows) * 40 // 100)]

        if to_compress:
            # Create summary
            summary = self._summarize_windows(to_compress)
            self.summaries.append(summary)

            # Remove compressed windows
            compressed_ids = {id(w) for w in to_compress}
            self.windows = [w for w in self.windows if id(w) not in compressed_ids]

            # Recalculate token count
            self.current_token_count = sum(w.token_estimate for w in self.windows)

    def _summarize_windows(self, windows: List[ContextWindow]) -> str:
        """Create a summary of multiple windows."""
        # Simple concatenation with markers (could use LLM for better summarization)
        contents = [w.content for w in windows]
        combined = " | ".join(contents[:5])  # Limit to first 5

        # Truncate if too long
        if len(combined) > 500:
            combined = combined[:497] + "..."

        time_range = f"{self._format_time_ago(windows[0].timestamp)} to {self._format_time_ago(windows[-1].timestamp)}"
        return f"[{time_range}] {combined}"

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (~ 4 chars per token)."""
        return len(text) // 4

    def _format_time_ago(self, timestamp: float) -> str:
        """Format timestamp as relative time."""
        age = time.time() - timestamp
        if age < 60:
            return f"{int(age)}s ago"
        elif age < 3600:
            return f"{int(age / 60)}m ago"
        else:
            return f"{age / 3600:.1f}h ago"

    def prioritize_by_recency(self, decay_factor: float = 0.9) -> None:
        """Adjust importance based on recency."""
        if not self.windows:
            return

        current_time = time.time()
        oldest = min(w.timestamp for w in self.windows)
        time_span = current_time - oldest

        if time_span == 0:
            return

        for window in self.windows:
            age = current_time - window.timestamp
            recency_score = 1.0 - (age / time_span)
            window.importance = window.importance * (1 - decay_factor) + recency_score * decay_factor

    def clear_old_context(self, max_age_seconds: float = 3600) -> int:
        """Remove context older than specified age."""
        cutoff = time.time() - max_age_seconds
        initial_count = len(self.windows)

        self.windows = [w for w in self.windows if w.timestamp > cutoff]
        removed = initial_count - len(self.windows)

        self.current_token_count = sum(w.token_estimate for w in self.windows)
        return removed

    def get_stats(self) -> Dict[str, Any]:
        """Get context manager statistics."""
        return {
            "total_windows": len(self.windows),
            "total_summaries": len(self.summaries),
            "current_tokens": self.current_token_count,
            "max_tokens": self.max_tokens,
            "utilization": self.current_token_count / self.max_tokens if self.max_tokens > 0 else 0,
            "oldest_window_age": time.time() - min((w.timestamp for w in self.windows), default=time.time()),
        }

    def format_status(self) -> str:
        """Format status as readable text."""
        stats = self.get_stats()
        lines = ["=== Context Manager Status ===\n"]
        lines.append(f"Windows: {stats['total_windows']}")
        lines.append(f"Summaries: {stats['total_summaries']}")
        lines.append(f"Tokens: {stats['current_tokens']} / {stats['max_tokens']} ({stats['utilization']:.1%})")
        if self.windows:
            lines.append(f"Oldest window: {stats['oldest_window_age']:.0f}s ago")
        return "\n".join(lines)
