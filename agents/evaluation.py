"""Agent evaluation metrics and benchmarking.

This module provides tools for evaluating agent performance including
success rates, latency, resource usage, and custom metrics.
"""
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
import statistics


@dataclass
class EvaluationResult:
    """Result from evaluating an agent on a single task."""
    task_id: str
    success: bool
    execution_time: float
    output: Any
    expected_output: Optional[Any] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "execution_time": self.execution_time,
            "output": str(self.output),
            "expected_output": str(self.expected_output) if self.expected_output else None,
            "score": self.score,
            "metadata": self.metadata
        }


@dataclass
class BenchmarkSuite:
    """A suite of test cases for evaluating agents."""
    name: str
    test_cases: List[Dict[str, Any]]
    scoring_fn: Optional[Callable] = None
    
    def add_test_case(self, task_id: str, input_data: Any, expected_output: Any = None) -> None:
        """Add a test case to the suite.
        
        Args:
            task_id: Unique identifier for the test
            input_data: Input to provide to the agent
            expected_output: Expected output (optional)
        """
        self.test_cases.append({
            "task_id": task_id,
            "input": input_data,
            "expected": expected_output
        })


class AgentEvaluator:
    """Evaluates agent performance across multiple metrics.
    
    Tracks:
    - Success rate
    - Execution time (mean, median, p95, p99)
    - Error rate and types
    - Custom scoring metrics
    """
    
    def __init__(self, agent_name: str = "agent"):
        """Initialize evaluator.
        
        Args:
            agent_name: Name of the agent being evaluated
        """
        self.agent_name = agent_name
        self.results: List[EvaluationResult] = []
        self.start_time: Optional[float] = None
    
    def evaluate_task(self, 
                     task_id: str,
                     agent_fn: Callable,
                     input_data: Any,
                     expected_output: Any = None,
                     scoring_fn: Optional[Callable] = None) -> EvaluationResult:
        """Evaluate agent on a single task.
        
        Args:
            task_id: Task identifier
            agent_fn: Agent function to evaluate (takes input, returns output)
            input_data: Input to provide
            expected_output: Expected output for comparison
            scoring_fn: Optional function to score the output (takes output, expected, returns float)
            
        Returns:
            EvaluationResult
        """
        start = time.time()
        success = False
        output = None
        score = None
        metadata = {}
        
        try:
            output = agent_fn(input_data)
            success = True
            
            # Calculate score if scoring function provided
            if scoring_fn and expected_output is not None:
                score = scoring_fn(output, expected_output)
            elif expected_output is not None:
                # Simple exact match scoring
                score = 1.0 if output == expected_output else 0.0
        except Exception as e:
            metadata["error"] = str(e)
            success = False
        
        execution_time = time.time() - start
        
        result = EvaluationResult(
            task_id=task_id,
            success=success,
            execution_time=execution_time,
            output=output,
            expected_output=expected_output,
            score=score,
            metadata=metadata
        )
        
        self.results.append(result)
        return result
    
    def evaluate_suite(self, 
                      suite: BenchmarkSuite,
                      agent_fn: Callable) -> List[EvaluationResult]:
        """Evaluate agent on a full benchmark suite.
        
        Args:
            suite: BenchmarkSuite with test cases
            agent_fn: Agent function to evaluate
            
        Returns:
            List of EvaluationResults
        """
        results = []
        for test_case in suite.test_cases:
            result = self.evaluate_task(
                task_id=test_case["task_id"],
                agent_fn=agent_fn,
                input_data=test_case["input"],
                expected_output=test_case.get("expected"),
                scoring_fn=suite.scoring_fn
            )
            results.append(result)
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate metrics from evaluation results.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.results:
            return {"error": "No evaluation results available"}
        
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total - successful
        
        # Execution times
        exec_times = [r.execution_time for r in self.results if r.success]
        
        # Scores (if available)
        scores = [r.score for r in self.results if r.score is not None]
        
        metrics = {
            "agent_name": self.agent_name,
            "total_tasks": total,
            "successful_tasks": successful,
            "failed_tasks": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "error_rate": failed / total if total > 0 else 0.0,
        }
        
        # Execution time stats
        if exec_times:
            metrics["execution_time"] = {
                "mean": statistics.mean(exec_times),
                "median": statistics.median(exec_times),
                "min": min(exec_times),
                "max": max(exec_times),
                "stdev": statistics.stdev(exec_times) if len(exec_times) > 1 else 0.0
            }
            
            # Percentiles
            sorted_times = sorted(exec_times)
            metrics["execution_time"]["p95"] = sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 0 else 0.0
            metrics["execution_time"]["p99"] = sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 0 else 0.0
        
        # Score stats
        if scores:
            metrics["scoring"] = {
                "mean_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "min_score": min(scores),
                "max_score": max(scores)
            }
        
        return metrics
    
    def get_failed_tasks(self) -> List[EvaluationResult]:
        """Get all failed task results.
        
        Returns:
            List of failed EvaluationResults
        """
        return [r for r in self.results if not r.success]
    
    def get_low_scoring_tasks(self, threshold: float = 0.5) -> List[EvaluationResult]:
        """Get tasks with scores below threshold.
        
        Args:
            threshold: Score threshold
            
        Returns:
            List of low-scoring EvaluationResults
        """
        return [r for r in self.results if r.score is not None and r.score < threshold]
    
    def reset(self) -> None:
        """Clear all evaluation results."""
        self.results = []
    
    def save_results(self, path: str) -> None:
        """Save evaluation results to JSON.
        
        Args:
            path: File path to save to
        """
        import json
        from pathlib import Path
        
        data = {
            "agent_name": self.agent_name,
            "results": [r.to_dict() for r in self.results],
            "metrics": self.get_metrics()
        }
        
        Path(path).write_text(json.dumps(data, indent=2))
    
    def __repr__(self) -> str:
        return f"AgentEvaluator(agent={self.agent_name}, results={len(self.results)})"


class ComparativeEvaluator:
    """Compare multiple agents on the same benchmark suite."""
    
    def __init__(self, suite: BenchmarkSuite):
        """Initialize comparative evaluator.
        
        Args:
            suite: BenchmarkSuite to use for all agents
        """
        self.suite = suite
        self.evaluators: Dict[str, AgentEvaluator] = {}
    
    def add_agent(self, agent_name: str, agent_fn: Callable) -> None:
        """Add an agent to compare.
        
        Args:
            agent_name: Name of the agent
            agent_fn: Agent function
        """
        evaluator = AgentEvaluator(agent_name)
        evaluator.evaluate_suite(self.suite, agent_fn)
        self.evaluators[agent_name] = evaluator
    
    def get_comparison(self) -> Dict[str, Any]:
        """Get comparison of all agents.
        
        Returns:
            Dictionary with comparative metrics
        """
        comparison = {
            "suite": self.suite.name,
            "agents": {}
        }
        
        for agent_name, evaluator in self.evaluators.items():
            comparison["agents"][agent_name] = evaluator.get_metrics()
        
        # Rankings
        if self.evaluators:
            # Rank by success rate
            by_success = sorted(
                self.evaluators.items(),
                key=lambda x: x[1].get_metrics().get("success_rate", 0),
                reverse=True
            )
            comparison["rankings"] = {
                "by_success_rate": [name for name, _ in by_success]
            }
            
            # Rank by speed (if available)
            agents_with_times = [
                (name, ev.get_metrics().get("execution_time", {}).get("mean", float('inf')))
                for name, ev in self.evaluators.items()
                if "execution_time" in ev.get_metrics()
            ]
            
            if agents_with_times:
                by_speed = sorted(agents_with_times, key=lambda x: x[1])
                comparison["rankings"]["by_speed"] = [name for name, _ in by_speed]
        
        return comparison
    
    def __repr__(self) -> str:
        return f"ComparativeEvaluator(suite={self.suite.name}, agents={len(self.evaluators)})"


def demo_evaluation():
    """Demonstrate agent evaluation."""
    print("=" * 70)
    print("Agent Evaluation Demo")
    print("=" * 70)
    
    # Define simple agent functions
    def fast_agent(input_data: str) -> str:
        """Fast but less accurate agent."""
        time.sleep(0.01)
        return input_data.upper()
    
    def slow_agent(input_data: str) -> str:
        """Slower but more accurate agent."""
        time.sleep(0.05)
        return input_data.upper() + "!"
    
    def buggy_agent(input_data: str) -> str:
        """Agent that sometimes fails."""
        if len(input_data) > 10:
            raise ValueError("Input too long")
        return input_data.lower()
    
    # Create benchmark suite
    suite = BenchmarkSuite(
        name="text_processing",
        test_cases=[],
        scoring_fn=lambda output, expected: 1.0 if output == expected else 0.5
    )
    
    suite.add_test_case("task1", "hello", "HELLO")
    suite.add_test_case("task2", "world", "WORLD")
    suite.add_test_case("task3", "test", "TEST")
    suite.add_test_case("task4", "short", "SHORT")
    suite.add_test_case("task5", "very long input text", "VERY LONG INPUT TEXT")
    
    # Evaluate single agent
    print("\n1. Single Agent Evaluation")
    print("-" * 70)
    
    evaluator = AgentEvaluator("fast_agent")
    evaluator.evaluate_suite(suite, fast_agent)
    
    metrics = evaluator.get_metrics()
    print(f"Agent: {metrics['agent_name']}")
    print(f"Success rate: {metrics['success_rate']:.2%}")
    print(f"Mean execution time: {metrics['execution_time']['mean']:.4f}s")
    print(f"P95 execution time: {metrics['execution_time']['p95']:.4f}s")
    
    # Comparative evaluation
    print("\n2. Comparative Evaluation (3 agents)")
    print("-" * 70)
    
    comp_eval = ComparativeEvaluator(suite)
    comp_eval.add_agent("fast_agent", fast_agent)
    comp_eval.add_agent("slow_agent", slow_agent)
    comp_eval.add_agent("buggy_agent", buggy_agent)
    
    comparison = comp_eval.get_comparison()
    
    print(f"\nRanked by success rate:")
    for i, agent_name in enumerate(comparison["rankings"]["by_success_rate"], 1):
        agent_metrics = comparison["agents"][agent_name]
        print(f"  {i}. {agent_name}: {agent_metrics['success_rate']:.2%}")
    
    print(f"\nRanked by speed:")
    for i, agent_name in enumerate(comparison["rankings"]["by_speed"], 1):
        agent_metrics = comparison["agents"][agent_name]
        mean_time = agent_metrics["execution_time"]["mean"]
        print(f"  {i}. {agent_name}: {mean_time:.4f}s")
    
    # Save results
    print("\n3. Saving Results")
    print("-" * 70)
    
    from pathlib import Path
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    result_path = "outputs/agent_evaluation.json"
    evaluator.save_results(result_path)
    print(f"Results saved to: {result_path}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_evaluation()
