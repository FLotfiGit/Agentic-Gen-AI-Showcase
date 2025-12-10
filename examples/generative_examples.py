"""Practical generative AI examples.

This module demonstrates real-world use cases combining prompt templates
and generation pipelines for various generative tasks.
"""
from generative_models.prompt_templates import TemplateRegistry
from generative_models.generation_pipeline import (
    GenerationPipeline, GenerationConfig,
    create_summarization_pipeline, create_code_pipeline, create_creative_pipeline,
    stub_generator, TrimWhitespaceStep, CapitalizeStep
)
from pathlib import Path
import json


# ============================================================================
# Example 1: Article Summarization
# ============================================================================

def example_article_summarization():
    """Demonstrate article summarization with different styles."""
    print("=" * 70)
    print("Example 1: Article Summarization")
    print("=" * 70)
    
    # Sample article
    article = """
    Artificial intelligence has made remarkable progress in recent years, particularly
    in the field of natural language processing. Large language models like GPT-3 and
    GPT-4 have demonstrated unprecedented capabilities in understanding and generating
    human-like text. These models are trained on vast amounts of data and can perform
    tasks ranging from translation to code generation. However, challenges remain,
    including concerns about bias, environmental impact, and the need for better
    interpretability. Researchers continue to work on making AI systems more reliable,
    efficient, and aligned with human values.
    """
    
    # Use template registry
    registry = TemplateRegistry()
    pipeline = create_summarization_pipeline()
    
    # Basic summary
    print("\nBasic Summary:")
    print("-" * 70)
    prompt = registry.format("summarize", text=article)
    result = pipeline.generate(prompt)
    print(result.text)
    
    # Bullet point summary
    print("\nBullet Point Summary:")
    print("-" * 70)
    bullet_prompt = registry.format("bullet_summary", text=article)
    result = pipeline.generate(bullet_prompt)
    print(result.text)
    
    # Technical style summary
    print("\nTechnical Summary:")
    print("-" * 70)
    tech_prompt = registry.format("summarize_with_style", text=article, style="technical")
    result = pipeline.generate(tech_prompt)
    print(result.text)
    print()


# ============================================================================
# Example 2: Question Answering System
# ============================================================================

def example_question_answering():
    """Demonstrate question answering with context."""
    print("=" * 70)
    print("Example 2: Question Answering System")
    print("=" * 70)
    
    # Knowledge base
    context = """
    Python was created by Guido van Rossum and first released in 1991. It is a
    high-level, interpreted programming language that emphasizes code readability
    with significant whitespace. Python supports multiple programming paradigms,
    including procedural, object-oriented, and functional programming. It has a
    comprehensive standard library and a large ecosystem of third-party packages
    available through PyPI (Python Package Index).
    """
    
    registry = TemplateRegistry()
    pipeline = GenerationPipeline(stub_generator)
    pipeline.add_preprocessing(TrimWhitespaceStep())
    pipeline.add_postprocessing(CapitalizeStep())
    
    # Multiple questions
    questions = [
        "Who created Python?",
        "When was Python first released?",
        "What programming paradigms does Python support?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 70)
        prompt = registry.format("qa", context=context, question=question)
        result = pipeline.generate(prompt)
        print(f"Answer: {result.text}\n")


# ============================================================================
# Example 3: Creative Story Generation
# ============================================================================

def example_story_generation():
    """Demonstrate creative story generation."""
    print("=" * 70)
    print("Example 3: Creative Story Generation")
    print("=" * 70)
    
    registry = TemplateRegistry()
    pipeline = create_creative_pipeline()
    
    # Story setup
    print("\nGenerating a science fiction story...")
    print("-" * 70)
    
    story_prompt = registry.format(
        "story_generation",
        genre="science fiction",
        setting="Mars colony in the year 2157",
        character="Captain Maya Rodriguez, a veteran pilot",
        conflict="mysterious signals from the Martian surface"
    )
    
    result = pipeline.generate(story_prompt)
    print(result.text)
    
    # Continue the story
    print("\n\nContinuing the story...")
    print("-" * 70)
    
    continue_prompt = registry.format(
        "continue_story",
        story_start=result.text
    )
    
    continuation = pipeline.generate(continue_prompt)
    print(continuation.text)
    print()


# ============================================================================
# Example 4: Code Documentation Generator
# ============================================================================

def example_code_documentation():
    """Demonstrate automatic code documentation."""
    print("=" * 70)
    print("Example 4: Code Documentation Generator")
    print("=" * 70)
    
    registry = TemplateRegistry()
    pipeline = create_code_pipeline()
    
    # Sample code
    code_samples = [
        {
            "language": "python",
            "code": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1"""
        },
        {
            "language": "javascript",
            "code": """function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}"""
        }
    ]
    
    for i, sample in enumerate(code_samples, 1):
        print(f"\nCode Sample {i}: {sample['language']}")
        print("-" * 70)
        
        # Generate documentation
        doc_prompt = registry.format(
            "code_documentation",
            language=sample["language"],
            code=sample["code"]
        )
        result = pipeline.generate(doc_prompt)
        print(f"Documentation:\n{result.text}")
        
        # Generate explanation
        explain_prompt = registry.format(
            "code_explanation",
            language=sample["language"],
            code=sample["code"]
        )
        result = pipeline.generate(explain_prompt)
        print(f"\nExplanation:\n{result.text}\n")


# ============================================================================
# Example 5: Multi-Language Translation
# ============================================================================

def example_translation():
    """Demonstrate text translation."""
    print("=" * 70)
    print("Example 5: Multi-Language Translation")
    print("=" * 70)
    
    registry = TemplateRegistry()
    pipeline = GenerationPipeline(stub_generator)
    
    text = "Machine learning is transforming the world."
    
    languages = [
        ("English", "Spanish"),
        ("English", "French"),
        ("English", "German")
    ]
    
    for source, target in languages:
        print(f"\n{source} → {target}:")
        print("-" * 70)
        prompt = registry.format(
            "translation",
            source_lang=source,
            target_lang=target,
            text=text
        )
        result = pipeline.generate(prompt)
        print(result.text)
    print()


# ============================================================================
# Example 6: Sentiment Analysis and Classification
# ============================================================================

def example_text_classification():
    """Demonstrate text classification and sentiment analysis."""
    print("=" * 70)
    print("Example 6: Text Classification & Sentiment Analysis")
    print("=" * 70)
    
    registry = TemplateRegistry()
    pipeline = GenerationPipeline(stub_generator)
    
    # Sentiment analysis
    print("\nSentiment Analysis:")
    print("-" * 70)
    
    texts = [
        "This product is absolutely amazing! Best purchase ever!",
        "Terrible experience, would not recommend.",
        "It's okay, nothing special but does the job."
    ]
    
    for text in texts:
        prompt = registry.format("sentiment", text=text)
        result = pipeline.generate(prompt)
        print(f"Text: {text[:50]}...")
        print(f"Sentiment: {result.text}\n")
    
    # Topic classification
    print("\nTopic Classification:")
    print("-" * 70)
    
    article = "The stock market saw significant gains today as tech companies reported strong earnings."
    categories = "Technology, Sports, Politics, Finance, Entertainment"
    
    prompt = registry.format("classification", categories=categories, text=article)
    result = pipeline.generate(prompt)
    print(f"Text: {article}")
    print(f"Category: {result.text}\n")


# ============================================================================
# Example 7: Batch Processing Pipeline
# ============================================================================

def example_batch_processing():
    """Demonstrate batch processing for efficiency."""
    print("=" * 70)
    print("Example 7: Batch Processing Pipeline")
    print("=" * 70)
    
    registry = TemplateRegistry()
    pipeline = create_summarization_pipeline()
    
    # Multiple articles to summarize
    articles = [
        "AI is revolutionizing healthcare with diagnostic tools.",
        "Climate change poses significant challenges for agriculture.",
        "Quantum computing promises to solve complex problems.",
        "Renewable energy adoption is accelerating globally.",
        "Space exploration enters a new commercial era."
    ]
    
    print("\nProcessing 5 articles in batch...")
    print("-" * 70)
    
    prompts = [registry.format("summarize", text=article) for article in articles]
    results = pipeline.batch_generate(prompts)
    
    for i, (article, result) in enumerate(zip(articles, results), 1):
        print(f"\n{i}. Original: {article}")
        print(f"   Summary: {result.text[:80]}...")
    
    print(f"\nTotal processing time: {sum(r.processing_time for r in results):.4f}s")
    print()


# ============================================================================
# Example 8: Save Results to File
# ============================================================================

def example_save_results():
    """Demonstrate saving generation results."""
    print("=" * 70)
    print("Example 8: Save Generation Results")
    print("=" * 70)
    
    registry = TemplateRegistry()
    pipeline = create_code_pipeline()
    
    # Generate code examples
    tasks = [
        "Create a function to validate email addresses",
        "Write a class for a binary search tree",
        "Implement a decorator for timing functions"
    ]
    
    results_data = []
    
    for task in tasks:
        prompt = registry.format("code_generation", language="python", task=task)
        result = pipeline.generate(prompt)
        
        results_data.append({
            "task": task,
            "prompt": result.prompt,
            "generated_code": result.text,
            "processing_time": result.processing_time,
            "metadata": result.metadata
        })
    
    # Save to file
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "generation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\nSaved {len(results_data)} generation results to: {output_file}")
    print("\nSample result:")
    print("-" * 70)
    print(f"Task: {results_data[0]['task']}")
    print(f"Generated:\n{results_data[0]['generated_code'][:150]}...")
    print()


# ============================================================================
# Main Demo Runner
# ============================================================================

def run_all_examples():
    """Run all generative AI examples."""
    print("\n" + "=" * 70)
    print(" " * 20 + "GENERATIVE AI EXAMPLES")
    print("=" * 70 + "\n")
    
    examples = [
        ("Article Summarization", example_article_summarization),
        ("Question Answering", example_question_answering),
        ("Story Generation", example_story_generation),
        ("Code Documentation", example_code_documentation),
        ("Translation", example_translation),
        ("Text Classification", example_text_classification),
        ("Batch Processing", example_batch_processing),
        ("Save Results", example_save_results),
    ]
    
    for name, example_func in examples:
        try:
            example_func()
            print()
        except Exception as e:
            print(f"Error in {name}: {e}\n")
    
    print("=" * 70)
    print(" " * 15 + "All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_examples()
