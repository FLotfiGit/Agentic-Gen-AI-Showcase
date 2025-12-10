"""Agent chains for composing multi-step workflows.

This module provides patterns for chaining agents together in sequential
and parallel execution patterns, enabling complex multi-agent workflows.
"""
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import time


@dataclass
class ChainStep:
    """A single step in an agent chain."""
    name: str
    function: Callable
    input_transform: Optional[Callable] = None
    output_transform: Optional[Callable] = None


@dataclass
class ChainResult:
    """Result from executing a chain."""
    success: bool
    outputs: List[Any]
    errors: List[str]
    execution_time: float
    metadata: Dict[str, Any]


class SequentialChain:
    """Execute agents/functions in sequence, passing output to next step.
    
    Each step receives the output of the previous step as input.
    Useful for linear workflows like: research → summarize → validate.
    """
    
    def __init__(self, steps: List[ChainStep], name: str = "sequential_chain"):
        """Initialize sequential chain.
        
        Args:
            steps: List of ChainStep objects to execute in order
            name: Chain name for logging
        """
        self.steps = steps
        self.name = name
        self.execution_history = []
    
    def run(self, initial_input: Any, stop_on_error: bool = True) -> ChainResult:
        """Execute the chain sequentially.
        
        Args:
            initial_input: Initial input to first step
            stop_on_error: Whether to stop execution on first error
            
        Returns:
            ChainResult with outputs from all steps
        """
        start_time = time.time()
        outputs = []
        errors = []
        current_input = initial_input
        
        for i, step in enumerate(self.steps):
            try:
                # Apply input transform if provided
                if step.input_transform:
                    current_input = step.input_transform(current_input)
                
                # Execute step
                output = step.function(current_input)
                
                # Apply output transform if provided
                if step.output_transform:
                    output = step.output_transform(output)
                
                outputs.append(output)
                current_input = output  # Pass to next step
                
                # Record execution
                self.execution_history.append({
                    "step": i,
                    "name": step.name,
                    "success": True,
                    "output_preview": str(output)[:100]
                })
                
            except Exception as e:
                error_msg = f"Step {i} ({step.name}) failed: {str(e)}"
                errors.append(error_msg)
                outputs.append(None)
                
                self.execution_history.append({
                    "step": i,
                    "name": step.name,
                    "success": False,
                    "error": str(e)
                })
                
                if stop_on_error:
                    break
        
        execution_time = time.time() - start_time
        success = len(errors) == 0
        
        return ChainResult(
            success=success,
            outputs=outputs,
            errors=errors,
            execution_time=execution_time,
            metadata={
                "chain_name": self.name,
                "steps_completed": len(outputs),
                "total_steps": len(self.steps)
            }
        )
    
    def __repr__(self) -> str:
        return f"SequentialChain(name={self.name}, steps={len(self.steps)})"


class ParallelChain:
    """Execute multiple agents/functions in parallel (simulated).
    
    All steps execute independently with the same input.
    Useful for: multiple retrievers, ensemble models, or diverse perspectives.
    
    Note: Currently uses sequential execution to simulate parallel.
    For true parallelism, use asyncio or threading in production.
    """
    
    def __init__(self, steps: List[ChainStep], name: str = "parallel_chain"):
        """Initialize parallel chain.
        
        Args:
            steps: List of ChainStep objects to execute in parallel
            name: Chain name for logging
        """
        self.steps = steps
        self.name = name
        self.execution_history = []
    
    def run(self, input_data: Any, fail_fast: bool = False) -> ChainResult:
        """Execute all steps with the same input (simulated parallel).
        
        Args:
            input_data: Input to provide to all steps
            fail_fast: Whether to stop on first error
            
        Returns:
            ChainResult with outputs from all steps
        """
        start_time = time.time()
        outputs = []
        errors = []
        
        for i, step in enumerate(self.steps):
            try:
                # Apply input transform if provided
                step_input = input_data
                if step.input_transform:
                    step_input = step.input_transform(step_input)
                
                # Execute step
                output = step.function(step_input)
                
                # Apply output transform if provided
                if step.output_transform:
                    output = step.output_transform(output)
                
                outputs.append(output)
                
                self.execution_history.append({
                    "step": i,
                    "name": step.name,
                    "success": True,
                    "output_preview": str(output)[:100]
                })
                
            except Exception as e:
                error_msg = f"Step {i} ({step.name}) failed: {str(e)}"
                errors.append(error_msg)
                outputs.append(None)
                
                self.execution_history.append({
                    "step": i,
                    "name": step.name,
                    "success": False,
                    "error": str(e)
                })
                
                if fail_fast:
                    break
        
        execution_time = time.time() - start_time
        success = len(errors) == 0
        
        return ChainResult(
            success=success,
            outputs=outputs,
            errors=errors,
            execution_time=execution_time,
            metadata={
                "chain_name": self.name,
                "steps_completed": len([o for o in outputs if o is not None]),
                "total_steps": len(self.steps)
            }
        )
    
    def __repr__(self) -> str:
        return f"ParallelChain(name={self.name}, steps={len(self.steps)})"


class ConditionalChain:
    """Execute different chains based on conditions.
    
    Routes execution to different sub-chains based on a condition function.
    Useful for: dynamic workflows, error recovery, or adaptive routing.
    """
    
    def __init__(self, name: str = "conditional_chain"):
        """Initialize conditional chain.
        
        Args:
            name: Chain name for logging
        """
        self.name = name
        self.routes: List[tuple] = []  # (condition_fn, chain) pairs
        self.default_chain: Optional[SequentialChain] = None
    
    def add_route(self, condition: Callable[[Any], bool], chain: SequentialChain) -> None:
        """Add a conditional route.
        
        Args:
            condition: Function that takes input and returns True/False
            chain: Chain to execute if condition is True
        """
        self.routes.append((condition, chain))
    
    def set_default(self, chain: SequentialChain) -> None:
        """Set default chain if no conditions match.
        
        Args:
            chain: Default chain to execute
        """
        self.default_chain = chain
    
    def run(self, input_data: Any) -> ChainResult:
        """Execute the appropriate chain based on conditions.
        
        Args:
            input_data: Input to evaluate and pass to selected chain
            
        Returns:
            ChainResult from the executed chain
        """
        # Find first matching route
        for condition, chain in self.routes:
            try:
                if condition(input_data):
                    return chain.run(input_data)
            except Exception as e:
                # Condition evaluation failed, try next
                continue
        
        # No condition matched, use default
        if self.default_chain:
            return self.default_chain.run(input_data)
        
        # No default chain
        return ChainResult(
            success=False,
            outputs=[],
            errors=["No matching route and no default chain"],
            execution_time=0.0,
            metadata={"chain_name": self.name}
        )
    
    def __repr__(self) -> str:
        return f"ConditionalChain(name={self.name}, routes={len(self.routes)})"


# Example usage functions
def example_research_function(topic: str) -> str:
    """Mock research function."""
    return f"Research findings about {topic}: [simulated data]"


def example_summarize_function(text: str) -> str:
    """Mock summarize function."""
    return f"Summary of '{text[:50]}...': [simulated summary]"


def example_validate_function(text: str) -> bool:
    """Mock validation function."""
    return len(text) > 10


def demo_chains():
    """Demonstrate different chain types."""
    print("=" * 70)
    print("Agent Chains Demo")
    print("=" * 70)
    
    # Sequential chain example
    print("\n1. Sequential Chain (research → summarize → validate)")
    print("-" * 70)
    
    sequential_steps = [
        ChainStep(name="research", function=example_research_function),
        ChainStep(name="summarize", function=example_summarize_function),
        ChainStep(name="validate", function=example_validate_function)
    ]
    
    seq_chain = SequentialChain(sequential_steps, name="research_pipeline")
    seq_result = seq_chain.run("artificial intelligence")
    
    print(f"Success: {seq_result.success}")
    print(f"Steps completed: {seq_result.metadata['steps_completed']}/{seq_result.metadata['total_steps']}")
    print(f"Execution time: {seq_result.execution_time:.3f}s")
    print(f"Final output: {seq_result.outputs[-1] if seq_result.outputs else 'None'}")
    
    # Parallel chain example
    print("\n2. Parallel Chain (multiple retrievers)")
    print("-" * 70)
    
    def retriever_a(query): return f"RetrieverA found: {query}"
    def retriever_b(query): return f"RetrieverB found: {query}"
    def retriever_c(query): return f"RetrieverC found: {query}"
    
    parallel_steps = [
        ChainStep(name="retriever_a", function=retriever_a),
        ChainStep(name="retriever_b", function=retriever_b),
        ChainStep(name="retriever_c", function=retriever_c)
    ]
    
    par_chain = ParallelChain(parallel_steps, name="ensemble_retrieval")
    par_result = par_chain.run("machine learning")
    
    print(f"Success: {par_result.success}")
    print(f"Results from {len(par_result.outputs)} retrievers:")
    for i, output in enumerate(par_result.outputs):
        print(f"  {i+1}. {output}")
    
    # Conditional chain example
    print("\n3. Conditional Chain (route based on input length)")
    print("-" * 70)
    
    short_chain = SequentialChain([
        ChainStep(name="short_handler", function=lambda x: f"Short input handler: {x}")
    ], name="short_workflow")
    
    long_chain = SequentialChain([
        ChainStep(name="long_handler", function=lambda x: f"Long input handler: {x[:50]}...")
    ], name="long_workflow")
    
    cond_chain = ConditionalChain(name="adaptive_router")
    cond_chain.add_route(lambda x: len(str(x)) < 20, short_chain)
    cond_chain.add_route(lambda x: len(str(x)) >= 20, long_chain)
    
    short_result = cond_chain.run("AI")
    long_result = cond_chain.run("This is a very long input text about artificial intelligence and machine learning")
    
    print(f"Short input: {short_result.outputs[0] if short_result.outputs else 'None'}")
    print(f"Long input: {long_result.outputs[0] if long_result.outputs else 'None'}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_chains()
