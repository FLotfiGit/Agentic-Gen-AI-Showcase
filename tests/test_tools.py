"""Tests for tool calling framework."""
import pytest
from agents.tools import (
    Tool, ToolResult, ToolRegistry,
    CalculatorTool, WebSearchTool, TextAnalysisTool,
    create_default_registry
)


def test_tool_result():
    """Test ToolResult creation."""
    result = ToolResult(success=True, output=42)
    assert result.success is True
    assert result.output == 42
    assert result.error is None
    
    # Failed result
    failed = ToolResult(success=False, output=None, error="Something went wrong")
    assert failed.success is False
    assert failed.error == "Something went wrong"


def test_calculator_tool():
    """Test calculator tool."""
    calc = CalculatorTool()
    
    assert calc.name == "calculator"
    assert "math" in calc.description.lower()
    
    # Basic arithmetic
    result = calc.execute(expression="2 + 2")
    assert result.success is True
    assert result.output == 4
    
    # Multiplication
    result = calc.execute(expression="10 * 5")
    assert result.success is True
    assert result.output == 50
    
    # Math functions
    result = calc.execute(expression="sqrt(16)")
    assert result.success is True
    assert result.output == 4.0
    
    # Invalid expression
    result = calc.execute(expression="invalid")
    assert result.success is False
    assert result.error is not None


def test_web_search_tool():
    """Test web search tool (simulated)."""
    search = WebSearchTool()
    
    assert search.name == "web_search"
    
    # Search with default results
    result = search.execute(query="AI trends")
    assert result.success is True
    assert isinstance(result.output, list)
    assert len(result.output) == 3  # Default num_results
    
    # Check result structure
    assert "title" in result.output[0]
    assert "snippet" in result.output[0]
    assert "url" in result.output[0]
    
    # Custom number of results
    result = search.execute(query="test", num_results=5)
    assert len(result.output) == 5


def test_text_analysis_tool():
    """Test text analysis tool."""
    analyzer = TextAnalysisTool()
    
    assert analyzer.name == "text_analysis"
    
    # Analyze simple text
    result = analyzer.execute(text="Hello world")
    assert result.success is True
    
    output = result.output
    assert output["word_count"] == 2
    assert output["character_count"] == 11
    assert output["character_count_no_spaces"] == 10
    
    # Test sentiment
    positive_result = analyzer.execute(text="This is great and wonderful!")
    assert positive_result.output["sentiment"] == "positive"
    
    negative_result = analyzer.execute(text="This is terrible and awful")
    assert negative_result.output["sentiment"] == "negative"
    
    neutral_result = analyzer.execute(text="The sky is blue")
    assert neutral_result.output["sentiment"] == "neutral"


def test_tool_registry():
    """Test tool registry."""
    registry = ToolRegistry()
    
    # Empty registry
    assert len(registry.list_tools()) == 0
    
    # Register tools
    calc = CalculatorTool()
    registry.register(calc)
    
    assert len(registry.list_tools()) == 1
    assert "calculator" in registry.list_tools()
    
    # Get tool
    retrieved = registry.get_tool("calculator")
    assert retrieved is not None
    assert retrieved.name == "calculator"
    
    # Get non-existent tool
    assert registry.get_tool("nonexistent") is None


def test_registry_execute():
    """Test executing tools via registry."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    
    # Execute tool
    result = registry.execute("calculator", expression="5 + 3")
    assert result.success is True
    assert result.output == 8
    
    # Execute non-existent tool
    result = registry.execute("nonexistent", arg="value")
    assert result.success is False
    assert "not found" in result.error.lower()


def test_registry_unregister():
    """Test unregistering tools."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    
    assert "calculator" in registry.list_tools()
    
    registry.unregister("calculator")
    assert "calculator" not in registry.list_tools()
    
    # Unregister non-existent tool (should not error)
    registry.unregister("nonexistent")


def test_default_registry():
    """Test default registry with built-in tools."""
    registry = create_default_registry()
    
    tools = registry.list_tools()
    assert "calculator" in tools
    assert "web_search" in tools
    assert "text_analysis" in tools
    
    # Verify all tools work
    calc_result = registry.execute("calculator", expression="10 / 2")
    assert calc_result.success is True
    assert calc_result.output == 5.0
    
    search_result = registry.execute("web_search", query="test")
    assert search_result.success is True
    
    analysis_result = registry.execute("text_analysis", text="test text")
    assert analysis_result.success is True


def test_tool_to_openai_function():
    """Test OpenAI function schema conversion."""
    calc = CalculatorTool()
    schema = calc.to_openai_function()
    
    assert "name" in schema
    assert "description" in schema
    assert "parameters" in schema
    
    assert schema["name"] == "calculator"
    assert "type" in schema["parameters"]
    assert schema["parameters"]["type"] == "object"


def test_registry_to_openai_functions():
    """Test converting all tools to OpenAI functions."""
    registry = create_default_registry()
    functions = registry.to_openai_functions()
    
    assert isinstance(functions, list)
    assert len(functions) == 3  # calculator, web_search, text_analysis
    
    # Check each function has required fields
    for func in functions:
        assert "name" in func
        assert "description" in func
        assert "parameters" in func


def test_calculator_advanced():
    """Test advanced calculator operations."""
    calc = CalculatorTool()
    
    # Exponentiation
    result = calc.execute(expression="2 ** 10")
    assert result.success is True
    assert result.output == 1024
    
    # Trigonometry
    result = calc.execute(expression="sin(0)")
    assert result.success is True
    assert abs(result.output) < 0.001  # Close to 0
    
    # Use pi constant
    result = calc.execute(expression="pi * 2")
    assert result.success is True
    assert abs(result.output - 6.283185307179586) < 0.001


def test_tool_result_to_dict():
    """Test converting ToolResult to dictionary."""
    result = ToolResult(
        success=True,
        output=42,
        metadata={"source": "test"}
    )
    
    result_dict = result.to_dict()
    assert result_dict["success"] is True
    assert result_dict["output"] == 42
    assert result_dict["metadata"]["source"] == "test"


def test_tool_error_handling():
    """Test tool error handling."""
    calc = CalculatorTool()
    
    # Division by zero
    result = calc.execute(expression="1 / 0")
    assert result.success is False
    assert "division" in result.error.lower() or "zero" in result.error.lower()
    
    # Undefined variable
    result = calc.execute(expression="undefined_var + 1")
    assert result.success is False
