"""Generation pipeline for composable text generation workflows.

This module provides a pipeline system for text generation with support for
preprocessing, generation, and post-processing steps.
"""
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import re
import time


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: List[str] = None
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []


@dataclass
class GenerationResult:
    """Result from text generation."""
    text: str
    prompt: str
    metadata: Dict[str, Any]
    processing_time: float
    
    def __str__(self) -> str:
        return self.text


class ProcessingStep:
    """Base class for pipeline processing steps."""
    
    def __init__(self, name: str):
        self.name = name
    
    def process(self, text: str, context: Dict[str, Any]) -> str:
        """Process text and return result.
        
        Args:
            text: Input text
            context: Shared context dictionary
            
        Returns:
            Processed text
        """
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


# ============================================================================
# Preprocessing Steps
# ============================================================================

class TrimWhitespaceStep(ProcessingStep):
    """Remove excess whitespace from text."""
    
    def __init__(self):
        super().__init__("trim_whitespace")
    
    def process(self, text: str, context: Dict[str, Any]) -> str:
        # Remove leading/trailing whitespace
        text = text.strip()
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text


class LowercaseStep(ProcessingStep):
    """Convert text to lowercase."""
    
    def __init__(self):
        super().__init__("lowercase")
    
    def process(self, text: str, context: Dict[str, Any]) -> str:
        return text.lower()


class TruncateStep(ProcessingStep):
    """Truncate text to maximum length."""
    
    def __init__(self, max_length: int = 1000):
        super().__init__("truncate")
        self.max_length = max_length
    
    def process(self, text: str, context: Dict[str, Any]) -> str:
        if len(text) > self.max_length:
            return text[:self.max_length] + "..."
        return text


class RemovePunctuationStep(ProcessingStep):
    """Remove punctuation from text."""
    
    def __init__(self):
        super().__init__("remove_punctuation")
    
    def process(self, text: str, context: Dict[str, Any]) -> str:
        return re.sub(r'[^\w\s]', '', text)


# ============================================================================
# Post-processing Steps
# ============================================================================

class ExtractFirstSentenceStep(ProcessingStep):
    """Extract only the first sentence."""
    
    def __init__(self):
        super().__init__("extract_first_sentence")
    
    def process(self, text: str, context: Dict[str, Any]) -> str:
        # Split on sentence boundaries
        match = re.search(r'^[^.!?]+[.!?]', text)
        if match:
            return match.group(0)
        return text


class CleanCodeBlocksStep(ProcessingStep):
    """Clean up code blocks in generated text."""
    
    def __init__(self):
        super().__init__("clean_code_blocks")
    
    def process(self, text: str, context: Dict[str, Any]) -> str:
        # Ensure code blocks are properly closed
        if '```' in text:
            count = text.count('```')
            if count % 2 != 0:
                # Odd number of backticks, add closing block
                text += '\n```'
        return text


class StopAtSequenceStep(ProcessingStep):
    """Stop generation at specific sequences."""
    
    def __init__(self, stop_sequences: List[str]):
        super().__init__("stop_at_sequence")
        self.stop_sequences = stop_sequences
    
    def process(self, text: str, context: Dict[str, Any]) -> str:
        for seq in self.stop_sequences:
            if seq in text:
                text = text[:text.index(seq)]
                break
        return text


class CapitalizeStep(ProcessingStep):
    """Capitalize the first letter."""
    
    def __init__(self):
        super().__init__("capitalize")
    
    def process(self, text: str, context: Dict[str, Any]) -> str:
        if text:
            return text[0].upper() + text[1:]
        return text


# ============================================================================
# Generation Pipeline
# ============================================================================

class GenerationPipeline:
    """Composable pipeline for text generation.
    
    Supports:
    - Preprocessing steps (applied before generation)
    - Generation function (LLM or stub)
    - Post-processing steps (applied after generation)
    - Context sharing between steps
    """
    
    def __init__(self, 
                 generator: Callable[[str, GenerationConfig], str],
                 config: Optional[GenerationConfig] = None,
                 name: str = "generation_pipeline"):
        """Initialize generation pipeline.
        
        Args:
            generator: Function that takes (prompt, config) and returns generated text
            config: Generation configuration
            name: Pipeline name
        """
        self.generator = generator
        self.config = config or GenerationConfig()
        self.name = name
        
        self.preprocessing_steps: List[ProcessingStep] = []
        self.postprocessing_steps: List[ProcessingStep] = []
        self.context: Dict[str, Any] = {}
    
    def add_preprocessing(self, step: ProcessingStep) -> 'GenerationPipeline':
        """Add a preprocessing step.
        
        Args:
            step: ProcessingStep to add
            
        Returns:
            Self for chaining
        """
        self.preprocessing_steps.append(step)
        return self
    
    def add_postprocessing(self, step: ProcessingStep) -> 'GenerationPipeline':
        """Add a post-processing step.
        
        Args:
            step: ProcessingStep to add
            
        Returns:
            Self for chaining
        """
        self.postprocessing_steps.append(step)
        return self
    
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Run the complete generation pipeline.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional context variables
            
        Returns:
            GenerationResult
        """
        start_time = time.time()
        
        # Update context with kwargs
        self.context.update(kwargs)
        self.context['original_prompt'] = prompt
        
        # Preprocessing
        processed_prompt = prompt
        for step in self.preprocessing_steps:
            processed_prompt = step.process(processed_prompt, self.context)
            self.context[f'after_{step.name}'] = processed_prompt
        
        # Generation
        generated_text = self.generator(processed_prompt, self.config)
        self.context['raw_generation'] = generated_text
        
        # Post-processing
        final_text = generated_text
        for step in self.postprocessing_steps:
            final_text = step.process(final_text, self.context)
            self.context[f'after_{step.name}'] = final_text
        
        processing_time = time.time() - start_time
        
        return GenerationResult(
            text=final_text,
            prompt=processed_prompt,
            metadata={
                'original_prompt': prompt,
                'preprocessing_steps': len(self.preprocessing_steps),
                'postprocessing_steps': len(self.postprocessing_steps),
                'config': self.config,
            },
            processing_time=processing_time
        )
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        """Generate for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional context variables
            
        Returns:
            List of GenerationResults
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]
    
    def __repr__(self) -> str:
        return f"GenerationPipeline(name={self.name}, pre={len(self.preprocessing_steps)}, post={len(self.postprocessing_steps)})"


# ============================================================================
# Stub Generator for Testing
# ============================================================================

def stub_generator(prompt: str, config: GenerationConfig) -> str:
    """Stub generator for testing without LLM API.
    
    Args:
        prompt: Input prompt
        config: Generation config
        
    Returns:
        Simulated generated text
    """
    # Simple rule-based generation for demo
    if "summarize" in prompt.lower():
        return "This is a concise summary of the provided text, capturing the main points and key information."
    elif "question" in prompt.lower() or "?" in prompt:
        return "Based on the provided context, the answer is that this is a simulated response for demonstration purposes."
    elif "code" in prompt.lower():
        return "```python\ndef example_function():\n    return 'Generated code example'\n```"
    elif "story" in prompt.lower():
        return "Once upon a time, in a world not unlike our own, something extraordinary happened. This is a simulated story generation."
    else:
        return "This is a generated response to the prompt. In production, this would be replaced by actual LLM output."


# ============================================================================
# Pre-built Pipelines
# ============================================================================

def create_summarization_pipeline(generator: Optional[Callable] = None) -> GenerationPipeline:
    """Create a pipeline optimized for summarization.
    
    Args:
        generator: Optional custom generator function
        
    Returns:
        Configured GenerationPipeline
    """
    gen = generator or stub_generator
    config = GenerationConfig(max_length=200, temperature=0.3)
    
    pipeline = GenerationPipeline(gen, config, name="summarization")
    pipeline.add_preprocessing(TrimWhitespaceStep())
    pipeline.add_postprocessing(TrimWhitespaceStep())
    pipeline.add_postprocessing(CapitalizeStep())
    
    return pipeline


def create_code_pipeline(generator: Optional[Callable] = None) -> GenerationPipeline:
    """Create a pipeline optimized for code generation.
    
    Args:
        generator: Optional custom generator function
        
    Returns:
        Configured GenerationPipeline
    """
    gen = generator or stub_generator
    config = GenerationConfig(max_length=1000, temperature=0.2)
    
    pipeline = GenerationPipeline(gen, config, name="code_generation")
    pipeline.add_preprocessing(TrimWhitespaceStep())
    pipeline.add_postprocessing(CleanCodeBlocksStep())
    
    return pipeline


def create_creative_pipeline(generator: Optional[Callable] = None) -> GenerationPipeline:
    """Create a pipeline optimized for creative writing.
    
    Args:
        generator: Optional custom generator function
        
    Returns:
        Configured GenerationPipeline
    """
    gen = generator or stub_generator
    config = GenerationConfig(max_length=800, temperature=0.9)
    
    pipeline = GenerationPipeline(gen, config, name="creative_writing")
    pipeline.add_preprocessing(TrimWhitespaceStep())
    pipeline.add_postprocessing(TrimWhitespaceStep())
    
    return pipeline


def demo_pipeline():
    """Demonstrate generation pipeline usage."""
    print("=" * 70)
    print("Generation Pipeline Demo")
    print("=" * 70)
    
    # Example 1: Basic pipeline
    print("\n1. Basic Pipeline with Preprocessing")
    print("-" * 70)
    
    pipeline = GenerationPipeline(stub_generator)
    pipeline.add_preprocessing(TrimWhitespaceStep())
    pipeline.add_postprocessing(CapitalizeStep())
    
    result = pipeline.generate("   summarize this text   ")
    print(f"Generated: {result.text}")
    print(f"Processing time: {result.processing_time:.4f}s")
    
    # Example 2: Summarization pipeline
    print("\n2. Pre-built Summarization Pipeline")
    print("-" * 70)
    
    sum_pipeline = create_summarization_pipeline()
    result = sum_pipeline.generate("Please summarize the main points of this lengthy article about AI.")
    print(f"Generated: {result.text}")
    
    # Example 3: Code generation pipeline
    print("\n3. Code Generation Pipeline")
    print("-" * 70)
    
    code_pipeline = create_code_pipeline()
    result = code_pipeline.generate("Write Python code to calculate fibonacci numbers")
    print(f"Generated:\n{result.text}")
    
    # Example 4: Custom pipeline with stop sequences
    print("\n4. Custom Pipeline with Stop Sequences")
    print("-" * 70)
    
    custom_pipeline = GenerationPipeline(
        stub_generator,
        config=GenerationConfig(stop_sequences=["END", "\n\n"])
    )
    custom_pipeline.add_postprocessing(StopAtSequenceStep(["END", "\n\n"]))
    
    result = custom_pipeline.generate("Generate a short response")
    print(f"Generated: {result.text}")
    
    # Example 5: Batch generation
    print("\n5. Batch Generation")
    print("-" * 70)
    
    prompts = [
        "Summarize AI trends",
        "Write code for sorting",
        "Tell a short story"
    ]
    
    batch_results = pipeline.batch_generate(prompts)
    for i, result in enumerate(batch_results, 1):
        print(f"{i}. {result.text[:80]}...")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_pipeline()
