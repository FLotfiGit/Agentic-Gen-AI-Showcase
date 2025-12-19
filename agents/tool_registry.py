from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any, Dict, List, Optional
import json


@dataclass
class ToolSignature:
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "required": self.required,
        }


@dataclass
class ToolResult:
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0


class Tool:
    """Base class for agentic tools."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: Optional[Dict[str, Any]] = None,
        required: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or {}
        self.required = required or []

    def signature(self) -> ToolSignature:
        return ToolSignature(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            required=self.required,
        )

    def validate_params(self, **kwargs) -> bool:
        """Check if all required parameters are present."""
        return all(req in kwargs for req in self.required)

    def execute(self, **kwargs) -> ToolResult:
        """Execute tool with parameters."""
        import time

        if not self.validate_params(**kwargs):
            missing = [r for r in self.required if r not in kwargs]
            return ToolResult(
                success=False,
                output=None,
                error=f"Missing required parameters: {missing}",
            )

        start = time.time()
        try:
            output = self.func(**kwargs)
            return ToolResult(
                success=True,
                output=output,
                execution_time=time.time() - start,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=time.time() - start,
            )


class ToolRegistry:
    """Central registry for agent tools."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: Optional[Dict[str, Any]] = None,
        required: Optional[List[str]] = None,
    ) -> Tool:
        """Register a new tool."""
        tool = Tool(name, description, func, parameters, required)
        self.tools[name] = tool
        return tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[ToolSignature]:
        """List all registered tools."""
        return [tool.signature() for tool in self.tools.values()]

    def describe_tools(self) -> str:
        """Get human-readable tool descriptions."""
        lines = ["Available Tools:\n"]
        for sig in self.list_tools():
            lines.append(f"• {sig.name}: {sig.description}")
            if sig.required:
                lines.append(f"  Required: {', '.join(sig.required)}")
        return "\n".join(lines)

    def execute_tool(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{name}' not found",
            )
        return tool.execute(**kwargs)

    def execute_batch(self, commands: List[Dict[str, Any]]) -> List[ToolResult]:
        """Execute multiple tools sequentially."""
        results = []
        for cmd in commands:
            name = cmd.get("name")
            params = cmd.get("params", {})
            result = self.execute_tool(name, **params)
            results.append(result)
        return results

    def to_json(self) -> str:
        """Export tool registry as JSON."""
        return json.dumps(
            {"tools": [sig.to_dict() for sig in self.list_tools()]},
            indent=2,
        )


# Built-in stub tools for demos
def create_default_tools() -> ToolRegistry:
    """Create a registry with useful stub tools."""
    registry = ToolRegistry()

    def search_web(query: str) -> str:
        return f"[STUB] Search results for '{query}': placeholder results..."

    def calculate(expression: str) -> float:
        try:
            return float(eval(expression))
        except:
            raise ValueError(f"Invalid expression: {expression}")

    def get_time() -> str:
        from datetime import datetime

        return datetime.now().isoformat()

    def summarize_text(text: str, max_length: int = 100) -> str:
        words = text.split()
        if len(words) <= max_length // 4:
            return text
        truncated = " ".join(words[: max_length // 4])
        return truncated + "..."

    registry.register(
        "search",
        "Search the web for information",
        search_web,
        parameters={"query": {"type": "string"}},
        required=["query"],
    )

    registry.register(
        "calculate",
        "Evaluate a mathematical expression",
        calculate,
        parameters={"expression": {"type": "string"}},
        required=["expression"],
    )

    registry.register(
        "get_time",
        "Get current time",
        get_time,
        parameters={},
        required=[],
    )

    registry.register(
        "summarize",
        "Summarize text to a target length",
        summarize_text,
        parameters={
            "text": {"type": "string"},
            "max_length": {"type": "integer", "default": 100},
        },
        required=["text"],
    )

    return registry
