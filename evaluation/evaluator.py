class Evaluator:
    """Simple evaluation stub for agents and generations."""
    def score_text(self, text: str) -> float:
        # naive length-based score
        return min(len(text) / 100.0, 1.0)

if __name__ == "__main__":
    e = Evaluator()
    sample = "Agentic reasoning improves adaptability."
    print("Score:", e.score_text(sample))
