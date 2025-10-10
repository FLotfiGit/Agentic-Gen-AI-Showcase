# Reasoning Agent Module Documentation

## Overview

The Reasoning Agent module implements the **ReAct (Reasoning + Acting)** pattern, which enables LLM-based agents to solve complex problems through iterative reasoning and action execution.

## Architecture

The module consists of the following key components:

### Core Classes

#### `ReasoningAgent`
The main agent class that orchestrates the reasoning loop.

**Key Methods:**
- `run(task, context)`: Execute the reasoning loop to complete a task
- `think(context, step)`: Generate a reasoning step
- `plan_action(thought)`: Plan the next action based on reasoning
- `execute_action(action)`: Execute a planned action
- `reflect(history)`: Reflect on progress and determine next steps

#### `Tool`
Represents a capability that the agent can use.

**Attributes:**
- `name`: Unique identifier for the tool
- `description`: What the tool does (used by LLM for selection)
- `function`: The callable function to execute

#### `Thought`
Represents a reasoning step in the agent's decision process.

**Attributes:**
- `content`: The reasoning text
- `step_number`: Position in the reasoning chain
- `thought_type`: Type of thought (reasoning, planning, reflection, observation)

#### `Action`
Represents an action the agent wants to take.

**Attributes:**
- `tool_name`: Name of the tool to use
- `parameters`: Dictionary of parameters for the tool
- `reasoning`: Why this action was chosen

## Usage

### Basic Example

```python
from modules.reasoning_agent import ReasoningAgent, Tool

# Define a custom tool
def calculator(expression: str) -> float:
    return eval(expression)

# Create tool
calc_tool = Tool(
    name="calculator",
    description="Performs mathematical calculations",
    function=calculator
)

# Create agent
agent = ReasoningAgent(tools=[calc_tool], max_iterations=5)

# Run task
result = agent.run(
    task="Calculate the result of (15 + 25) * 2",
    context="Use the calculator tool."
)

# Examine results
print(f"Status: {result['status']}")
print(f"Steps: {result['steps']}")
for entry in result['history']:
    print(f"{entry['type']}: {entry.get('content', '')}")
```

### Creating Custom Tools

```python
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information"""
    # Implementation here
    return f"Results for: {query}"

search_tool = Tool(
    name="wikipedia",
    description="Search Wikipedia for factual information",
    function=search_wikipedia
)

agent.add_tool(search_tool)
```

## ReAct Pattern

The agent follows this pattern:

1. **Thought**: Analyze the current situation and reason about what to do
2. **Action**: Decide which tool to use and with what parameters
3. **Observation**: Process the result from the tool execution
4. **Repeat**: Continue until the task is complete or max iterations reached

### Example Flow

```
Task: "What is the capital of France and what is its population?"

Thought 1: I need to find information about France's capital
Action 1: search("capital of France")
Observation 1: Paris is the capital of France

Thought 2: Now I need to find the population of Paris
Action 2: search("population of Paris")
Observation 2: Paris has approximately 2.2 million inhabitants

Thought 3: I now have all the information needed
Final Answer: The capital of France is Paris, with a population of approximately 2.2 million.
```

## Integration with LLMs

To integrate with real LLMs (OpenAI, Anthropic, etc.):

1. **Modify `think()` method**: Call LLM API with context and prompt template
2. **Modify `plan_action()` method**: Parse LLM response to extract action
3. **Use the prompt template**: Available via `get_prompt_template()`

### Example LLM Integration (Pseudocode)

```python
def think(self, context: str, step: int) -> Thought:
    prompt = self.get_prompt_template().format(
        task=context,
        context=self._format_history()
    )
    
    # Call LLM
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return Thought(
        content=response.choices[0].message.content,
        step_number=step,
        thought_type='reasoning'
    )
```

## Best Practices

1. **Tool Descriptions**: Make tool descriptions clear and specific
2. **Error Handling**: Implement robust error handling in tool functions
3. **Max Iterations**: Set appropriate limits to prevent infinite loops
4. **Context Management**: Keep context focused and relevant
5. **Reflection**: Use reflection to validate progress and adjust strategy

## Advanced Features

### Multi-step Planning

```python
# Agent can plan multiple steps ahead
result = agent.run(
    task="Book a flight and hotel for a trip to Paris",
    context="Available tools: search_flights, search_hotels, make_booking"
)
```

### Self-Correction

The agent can reflect on errors and adjust its approach:

```python
# If a tool fails, the agent can try alternative approaches
# Implemented in the reflect() method
```

## Limitations

- Current implementation uses placeholder reasoning (needs LLM integration)
- Tool execution is synchronous (could be made parallel)
- No persistent memory between runs
- Limited error recovery strategies

## Future Enhancements

- [ ] Integration with major LLM providers (OpenAI, Anthropic, etc.)
- [ ] Parallel tool execution
- [ ] Persistent memory and learning
- [ ] Multi-agent collaboration
- [ ] Tool discovery and dynamic loading
- [ ] Advanced planning algorithms (MCTS, beam search)

## References

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
