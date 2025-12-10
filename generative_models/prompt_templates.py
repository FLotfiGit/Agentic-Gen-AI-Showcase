"""Prompt template system for generative AI tasks.

This module provides reusable prompt templates and formatters for common
generative tasks including summarization, Q&A, creative writing, and code generation.
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from string import Template


@dataclass
class PromptTemplate:
    """A reusable prompt template with variable substitution."""
    name: str
    template: str
    input_variables: List[str]
    description: str = ""
    
    def format(self, **kwargs) -> str:
        """Format the template with provided variables.
        
        Args:
            **kwargs: Variable values to substitute
            
        Returns:
            Formatted prompt string
        """
        # Check all required variables are provided
        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        # Use string Template for safe substitution
        tmpl = Template(self.template)
        return tmpl.safe_substitute(**kwargs)
    
    def partial(self, **kwargs) -> 'PromptTemplate':
        """Create a new template with some variables pre-filled.
        
        Args:
            **kwargs: Variable values to pre-fill
            
        Returns:
            New PromptTemplate with partial values
        """
        # Substitute provided values
        tmpl = Template(self.template)
        new_template = tmpl.safe_substitute(**kwargs)
        
        # Update input variables list
        remaining_vars = [v for v in self.input_variables if v not in kwargs]
        
        return PromptTemplate(
            name=f"{self.name}_partial",
            template=new_template,
            input_variables=remaining_vars,
            description=f"Partial: {self.description}"
        )


# ============================================================================
# Summarization Templates
# ============================================================================

SUMMARIZE_TEMPLATE = PromptTemplate(
    name="summarize",
    template="""Summarize the following text in a concise manner:

Text: $text

Summary:""",
    input_variables=["text"],
    description="Basic text summarization"
)

SUMMARIZE_WITH_STYLE_TEMPLATE = PromptTemplate(
    name="summarize_with_style",
    template="""Summarize the following text in a $style style:

Text: $text

$style summary:""",
    input_variables=["text", "style"],
    description="Summarization with specific style (e.g., technical, casual, formal)"
)

BULLET_SUMMARY_TEMPLATE = PromptTemplate(
    name="bullet_summary",
    template="""Create a bullet-point summary of the following text:

Text: $text

Key points:
-""",
    input_variables=["text"],
    description="Bullet-point summary format"
)


# ============================================================================
# Question Answering Templates
# ============================================================================

QA_TEMPLATE = PromptTemplate(
    name="qa",
    template="""Answer the following question based on the context provided.

Context: $context

Question: $question

Answer:""",
    input_variables=["context", "question"],
    description="Basic question answering with context"
)

QA_WITH_CITATION_TEMPLATE = PromptTemplate(
    name="qa_with_citation",
    template="""Answer the following question based on the context provided. Include citations to the relevant parts of the context.

Context: $context

Question: $question

Answer (with citations):""",
    input_variables=["context", "question"],
    description="Question answering with source citations"
)

MULTI_CONTEXT_QA_TEMPLATE = PromptTemplate(
    name="multi_context_qa",
    template="""Answer the following question based on the multiple sources provided.

Sources:
$sources

Question: $question

Answer:""",
    input_variables=["sources", "question"],
    description="Question answering from multiple sources"
)


# ============================================================================
# Creative Writing Templates
# ============================================================================

STORY_GENERATION_TEMPLATE = PromptTemplate(
    name="story_generation",
    template="""Write a $genre story with the following elements:

Setting: $setting
Main character: $character
Conflict: $conflict

Story:""",
    input_variables=["genre", "setting", "character", "conflict"],
    description="Generate creative stories with specific elements"
)

CONTINUE_STORY_TEMPLATE = PromptTemplate(
    name="continue_story",
    template="""Continue the following story in a natural and engaging way:

Story so far:
$story_start

Continue the story:""",
    input_variables=["story_start"],
    description="Continue an existing story"
)

DIALOGUE_TEMPLATE = PromptTemplate(
    name="dialogue",
    template="""Write a dialogue between $character1 and $character2 about $topic.

Dialogue:""",
    input_variables=["character1", "character2", "topic"],
    description="Generate dialogue between characters"
)


# ============================================================================
# Code Generation Templates
# ============================================================================

CODE_GENERATION_TEMPLATE = PromptTemplate(
    name="code_generation",
    template="""Write $language code to accomplish the following task:

Task: $task

Code:
```$language""",
    input_variables=["language", "task"],
    description="Generate code in specified language"
)

CODE_DOCUMENTATION_TEMPLATE = PromptTemplate(
    name="code_documentation",
    template="""Write comprehensive documentation for the following code:

Code:
```$language
$code
```

Documentation:""",
    input_variables=["language", "code"],
    description="Generate documentation for code"
)

CODE_EXPLANATION_TEMPLATE = PromptTemplate(
    name="code_explanation",
    template="""Explain the following code in simple terms:

Code:
```$language
$code
```

Explanation:""",
    input_variables=["language", "code"],
    description="Explain code in plain language"
)

CODE_REVIEW_TEMPLATE = PromptTemplate(
    name="code_review",
    template="""Review the following code and provide constructive feedback:

Code:
```$language
$code
```

Review:
- Strengths:
- Areas for improvement:
- Suggestions:""",
    input_variables=["language", "code"],
    description="Code review with structured feedback"
)


# ============================================================================
# Translation Templates
# ============================================================================

TRANSLATION_TEMPLATE = PromptTemplate(
    name="translation",
    template="""Translate the following text from $source_lang to $target_lang:

Text: $text

Translation:""",
    input_variables=["source_lang", "target_lang", "text"],
    description="Language translation"
)


# ============================================================================
# Classification Templates
# ============================================================================

CLASSIFICATION_TEMPLATE = PromptTemplate(
    name="classification",
    template="""Classify the following text into one of these categories: $categories

Text: $text

Category:""",
    input_variables=["categories", "text"],
    description="Text classification"
)

SENTIMENT_TEMPLATE = PromptTemplate(
    name="sentiment",
    template="""Analyze the sentiment of the following text. Respond with: positive, negative, or neutral.

Text: $text

Sentiment:""",
    input_variables=["text"],
    description="Sentiment analysis"
)


# ============================================================================
# Template Registry
# ============================================================================

class TemplateRegistry:
    """Registry for managing prompt templates."""
    
    def __init__(self):
        """Initialize registry with default templates."""
        self.templates: Dict[str, PromptTemplate] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register all default templates."""
        default_templates = [
            # Summarization
            SUMMARIZE_TEMPLATE,
            SUMMARIZE_WITH_STYLE_TEMPLATE,
            BULLET_SUMMARY_TEMPLATE,
            # Q&A
            QA_TEMPLATE,
            QA_WITH_CITATION_TEMPLATE,
            MULTI_CONTEXT_QA_TEMPLATE,
            # Creative
            STORY_GENERATION_TEMPLATE,
            CONTINUE_STORY_TEMPLATE,
            DIALOGUE_TEMPLATE,
            # Code
            CODE_GENERATION_TEMPLATE,
            CODE_DOCUMENTATION_TEMPLATE,
            CODE_EXPLANATION_TEMPLATE,
            CODE_REVIEW_TEMPLATE,
            # Other
            TRANSLATION_TEMPLATE,
            CLASSIFICATION_TEMPLATE,
            SENTIMENT_TEMPLATE,
        ]
        
        for template in default_templates:
            self.templates[template.name] = template
    
    def register(self, template: PromptTemplate) -> None:
        """Register a custom template.
        
        Args:
            template: PromptTemplate to register
        """
        self.templates[template.name] = template
    
    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name.
        
        Args:
            name: Template name
            
        Returns:
            PromptTemplate or None if not found
        """
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List all registered template names.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
    
    def format(self, name: str, **kwargs) -> str:
        """Format a template by name.
        
        Args:
            name: Template name
            **kwargs: Variables to substitute
            
        Returns:
            Formatted prompt string
        """
        template = self.get(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        return template.format(**kwargs)
    
    def __repr__(self) -> str:
        return f"TemplateRegistry(templates={len(self.templates)})"


def demo_templates():
    """Demonstrate prompt template usage."""
    print("=" * 70)
    print("Prompt Template System Demo")
    print("=" * 70)
    
    registry = TemplateRegistry()
    
    # Example 1: Summarization
    print("\n1. Summarization Template")
    print("-" * 70)
    
    summary_prompt = registry.format(
        "summarize",
        text="Artificial intelligence is transforming industries worldwide. "
             "From healthcare to finance, AI systems are enabling new capabilities "
             "and efficiencies that were previously impossible."
    )
    print(summary_prompt)
    
    # Example 2: Q&A with context
    print("\n2. Question Answering Template")
    print("-" * 70)
    
    qa_prompt = registry.format(
        "qa",
        context="Python is a high-level programming language created by Guido van Rossum in 1991. "
                "It emphasizes code readability and simplicity.",
        question="Who created Python?"
    )
    print(qa_prompt)
    
    # Example 3: Story generation
    print("\n3. Creative Story Template")
    print("-" * 70)
    
    story_prompt = registry.format(
        "story_generation",
        genre="science fiction",
        setting="a space station orbiting Mars in 2150",
        character="Dr. Sarah Chen, a robotics engineer",
        conflict="the station's AI begins exhibiting unexpected behavior"
    )
    print(story_prompt[:200] + "...")
    
    # Example 4: Code documentation
    print("\n4. Code Documentation Template")
    print("-" * 70)
    
    code_doc_prompt = registry.format(
        "code_documentation",
        language="python",
        code="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    )
    print(code_doc_prompt[:200] + "...")
    
    # Example 5: Partial template
    print("\n5. Partial Template (Pre-filled Variables)")
    print("-" * 70)
    
    python_code_template = CODE_GENERATION_TEMPLATE.partial(language="python")
    python_prompt = python_code_template.format(task="sort a list of dictionaries by a key")
    print(python_prompt[:200] + "...")
    
    # List all templates
    print("\n6. Available Templates")
    print("-" * 70)
    print(f"Total templates: {len(registry.list_templates())}")
    for i, name in enumerate(registry.list_templates(), 1):
        template = registry.get(name)
        print(f"  {i}. {name}: {template.description}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_templates()
