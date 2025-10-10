"""
Reasoning-Driven LLM Agent Module

This module implements a reasoning-driven agent using the ReAct (Reasoning + Acting) pattern.
The agent can reason about problems, plan actions, and execute them in an iterative loop.

Key Features:
- Chain-of-thought reasoning
- Tool-use capabilities
- Self-reflection and error correction
- Planning and multi-step reasoning
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import json


@dataclass
class Thought:
    """Represents a reasoning step in the agent's decision process"""
    content: str
    step_number: int
    thought_type: str  # 'reasoning', 'planning', 'reflection', 'observation'


@dataclass
class Action:
    """Represents an action the agent wants to take"""
    tool_name: str
    parameters: Dict[str, Any]
    reasoning: str


class Tool:
    """Base class for tools that the agent can use"""
    
    def __init__(self, name: str, description: str, function: Callable):
        self.name = name
        self.description = description
        self.function = function
    
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        try:
            result = self.function(**kwargs)
            return str(result)
        except Exception as e:
            return f"Error executing {self.name}: {str(e)}"


class ReasoningAgent:
    """
    A reasoning-driven agent that uses the ReAct pattern for problem-solving.
    
    The agent follows this loop:
    1. Thought: Reason about the current situation
    2. Action: Decide what action to take
    3. Observation: Process the result of the action
    4. Repeat until the task is complete
    """
    
    def __init__(self, tools: Optional[List[Tool]] = None, max_iterations: int = 10):
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.max_iterations = max_iterations
        self.history: List[Dict[str, Any]] = []
        
    def add_tool(self, tool: Tool):
        """Add a new tool to the agent's toolkit"""
        self.tools[tool.name] = tool
    
    def think(self, context: str, step: int) -> Thought:
        """
        Generate a reasoning step based on current context.
        In a real implementation, this would call an LLM.
        """
        # This is a placeholder - in production, call an LLM API
        thought = Thought(
            content=f"Analyzing: {context[:100]}...",
            step_number=step,
            thought_type='reasoning'
        )
        return thought
    
    def plan_action(self, thought: Thought) -> Optional[Action]:
        """
        Based on the current thought, plan the next action.
        In a real implementation, this would call an LLM to decide the action.
        """
        # This is a placeholder - in production, use LLM to decide action
        if self.tools:
            tool_name = list(self.tools.keys())[0]
            return Action(
                tool_name=tool_name,
                parameters={},
                reasoning=thought.content
            )
        return None
    
    def execute_action(self, action: Action) -> str:
        """Execute the planned action using the appropriate tool"""
        if action.tool_name not in self.tools:
            return f"Error: Tool '{action.tool_name}' not found"
        
        tool = self.tools[action.tool_name]
        result = tool.execute(**action.parameters)
        return result
    
    def reflect(self, history: List[Dict[str, Any]]) -> str:
        """
        Reflect on the action history to determine if the task is complete
        or if adjustments are needed.
        """
        # This is a placeholder - in production, use LLM for reflection
        if len(history) >= self.max_iterations:
            return "Maximum iterations reached"
        return "Continue"
    
    def run(self, task: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the reasoning loop to complete the given task.
        
        Args:
            task: The task description
            context: Optional additional context
            
        Returns:
            Dictionary containing the execution history and final result
        """
        self.history = []
        current_context = f"Task: {task}\n{context or ''}"
        
        for step in range(self.max_iterations):
            # Think
            thought = self.think(current_context, step)
            self.history.append({
                'step': step,
                'type': 'thought',
                'content': thought.content
            })
            
            # Plan action
            action = self.plan_action(thought)
            if action is None:
                self.history.append({
                    'step': step,
                    'type': 'conclusion',
                    'content': 'No more actions needed'
                })
                break
            
            self.history.append({
                'step': step,
                'type': 'action',
                'tool': action.tool_name,
                'parameters': action.parameters
            })
            
            # Execute action
            observation = self.execute_action(action)
            self.history.append({
                'step': step,
                'type': 'observation',
                'content': observation
            })
            
            current_context += f"\nObservation: {observation}"
            
            # Reflect
            reflection = self.reflect(self.history)
            if reflection != "Continue":
                self.history.append({
                    'step': step,
                    'type': 'reflection',
                    'content': reflection
                })
                break
        
        return {
            'task': task,
            'history': self.history,
            'status': 'completed',
            'steps': len(self.history)
        }
    
    def get_prompt_template(self) -> str:
        """
        Returns the prompt template for the ReAct pattern.
        This would be used with an actual LLM.
        """
        tools_description = "\n".join([
            f"- {name}: {tool.description}" 
            for name, tool in self.tools.items()
        ])
        
        template = f"""You are a reasoning agent that solves problems step by step.

Available Tools:
{tools_description}

Follow this pattern for each step:
Thought: [Your reasoning about what to do next]
Action: [The tool to use]
Action Input: [The input to the tool]
Observation: [The result from the tool]
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: [Your final answer to the task]

Begin!

Task: {{task}}
{{context}}
"""
        return template


# Example tools for demonstration
def calculator(expression: str) -> float:
    """Simple calculator tool"""
    try:
        # Safe evaluation of basic math expressions
        allowed_chars = set('0123456789+-*/(). ')
        if all(c in allowed_chars for c in expression):
            return eval(expression)
        else:
            return "Invalid expression"
    except Exception as e:
        return f"Error: {str(e)}"


def search(query: str) -> str:
    """Mock search tool - in production, this would call a real search API"""
    return f"Search results for '{query}': [Mock results - implement with real API]"


def get_example_tools() -> List[Tool]:
    """Returns example tools for the reasoning agent"""
    return [
        Tool(
            name="calculator",
            description="Performs mathematical calculations. Input should be a valid mathematical expression.",
            function=calculator
        ),
        Tool(
            name="search",
            description="Searches for information. Input should be a search query string.",
            function=search
        )
    ]


if __name__ == "__main__":
    # Example usage
    print("=== Reasoning Agent Demo ===\n")
    
    # Create agent with example tools
    agent = ReasoningAgent(tools=get_example_tools(), max_iterations=5)
    
    # Run a simple task
    result = agent.run(
        task="Calculate the result of (15 + 25) * 2",
        context="Use the calculator tool to solve this problem."
    )
    
    print("Task:", result['task'])
    print("\nExecution History:")
    for entry in result['history']:
        print(f"  Step {entry['step']}: {entry['type']} - {entry.get('content', entry.get('tool', 'N/A'))}")
    
    print(f"\nStatus: {result['status']}")
    print(f"Total steps: {result['steps']}")
