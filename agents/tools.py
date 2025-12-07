"""Tool calling framework for agentic AI.

This module provides a base class for tools/functions that agents can call,
along with a registry to manage available tools.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
import json
import math
import re


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        if self.success:
            return f"Success: {self.output}"
        return f"Error: {self.error}"


class Tool(ABC):
    """Base class for agent tools.
    
    Each tool should implement:
    - name: unique tool identifier
    - description: what the tool does
    - parameters: dict describing expected parameters
    - execute: the actual tool logic
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Parameter schema (can follow OpenAI function calling format)."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters.
        
        Returns:
            ToolResult containing success status and output
        """
        pass
    
    def to_openai_function(self) -> Dict[str, Any]:
        """Format tool as OpenAI function calling schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    def __repr__(self) -> str:
        return f"Tool({self.name})"


class CalculatorTool(Tool):
    """Simple calculator tool for basic math operations."""
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return "Performs basic mathematical calculations. Supports +, -, *, /, **, sqrt, sin, cos, tan, log, exp."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(3.14159/2)')"
                }
            },
            "required": ["expression"]
        }
    
    def execute(self, expression: str, **kwargs) -> ToolResult:
        """Evaluate a mathematical expression safely.
        
        Args:
            expression: Math expression string
            
        Returns:
            ToolResult with calculated value or error
        """
        try:
            # Safe evaluation - only allow math operations
            allowed_names = {
                'sqrt': math.sqrt,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'log': math.log,
                'exp': math.exp,
                'pi': math.pi,
                'e': math.e
            }
            
            # Clean the expression
            expr = expression.strip()
            
            # Compile and evaluate with restricted globals
            code = compile(expr, "<string>", "eval")
            result = eval(code, {"__builtins__": {}}, allowed_names)
            
            return ToolResult(success=True, output=result)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class WebSearchTool(Tool):
    """Simulated web search tool (returns mock results)."""
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web for information. Returns simulated search results."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    
    def execute(self, query: str, num_results: int = 3, **kwargs) -> ToolResult:
        """Simulate a web search (returns mock data).
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            ToolResult with simulated search results
        """
        # Mock search results
        results = [
            {
                "title": f"Result {i+1} for '{query}'",
                "snippet": f"This is a simulated search result snippet about {query}. "
                          f"In a real implementation, this would query an actual search API.",
                "url": f"https://example.com/result{i+1}"
            }
            for i in range(min(num_results, 5))
        ]
        
        return ToolResult(
            success=True,
            output=results,
            metadata={"query": query, "num_results": len(results)}
        )


class TextAnalysisTool(Tool):
    """Analyze text properties (word count, sentiment, etc.)."""
    
    @property
    def name(self) -> str:
        return "text_analysis"
    
    @property
    def description(self) -> str:
        return "Analyze text to extract word count, character count, and basic sentiment."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to analyze"
                }
            },
            "required": ["text"]
        }
    
    def execute(self, text: str, **kwargs) -> ToolResult:
        """Analyze text properties.
        
        Args:
            text: Text to analyze
            
        Returns:
            ToolResult with analysis
        """
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        char_count_no_spaces = len(text.replace(" ", ""))
        
        # Simple sentiment (count positive/negative words)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'happy', 'love']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'sad', 'hate', 'angry']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        sentiment = "neutral"
        if pos_count > neg_count:
            sentiment = "positive"
        elif neg_count > pos_count:
            sentiment = "negative"
        
        analysis = {
            "word_count": word_count,
            "character_count": char_count,
            "character_count_no_spaces": char_count_no_spaces,
            "sentiment": sentiment,
            "sentiment_scores": {
                "positive": pos_count,
                "negative": neg_count
            }
        }
        
        return ToolResult(success=True, output=analysis)


class ToolRegistry:
    """Registry to manage available tools for agents."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool.
        
        Args:
            tool: Tool instance to register
        """
        self.tools[tool.name] = tool
    
    def unregister(self, tool_name: str) -> None:
        """Unregister a tool by name.
        
        Args:
            tool_name: Name of tool to remove
        """
        self.tools.pop(tool_name, None)
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name.
        
        Args:
            tool_name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())
    
    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name.
        
        Args:
            tool_name: Name of tool to execute
            **kwargs: Tool parameters
            
        Returns:
            ToolResult
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{tool_name}' not found"
            )
        
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool execution error: {str(e)}"
            )
    
    def to_openai_functions(self) -> List[Dict[str, Any]]:
        """Format all tools as OpenAI function calling schema.
        
        Returns:
            List of function schemas
        """
        return [tool.to_openai_function() for tool in self.tools.values()]
    
    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self.tools.keys())})"


# Create a default registry with basic tools
def create_default_registry() -> ToolRegistry:
    """Create a registry with default tools.
    
    Returns:
        ToolRegistry with calculator, web_search, and text_analysis tools
    """
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool())
    registry.register(TextAnalysisTool())
    return registry
