# Agent Architecture Documentation

This document provides a comprehensive overview of the agentic AI framework architecture, components, and usage patterns.

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Architecture Patterns](#architecture-patterns)
4. [Usage Examples](#usage-examples)
5. [Best Practices](#best-practices)

## Overview

The agentic AI framework provides a modular, extensible system for building intelligent agents with capabilities including:

- **Conversation Management**: Multi-turn dialogue tracking with context windowing
- **Tool Calling**: Extensible tool system for external function execution
- **State Management**: Stateful context tracking across steps and sessions
- **Chain Composition**: Sequential, parallel, and conditional agent workflows
- **Callbacks**: Lifecycle event monitoring and logging
- **Evaluation**: Performance metrics and benchmarking

### Key Design Principles

- **Modularity**: Each component is independent and composable
- **Extensibility**: Easy to add custom tools, agents, and workflows
- **Offline-first**: Deterministic stubs for development without API keys
- **Production-ready**: Designed to integrate with LLM APIs (OpenAI, Anthropic, etc.)

## Core Components

### 1. Agent Utilities (`agents/agent_utils.py`)

Basic agent primitives for plan-act loops.

**Key Classes:**
- `Thought`: Represents an agent thought/reasoning step
- `Action`: Represents an executable action
- `SimpleAgent`: Basic agent with plan→act→reflect loop

**Example:**
```python
from agents.agent_utils import SimpleAgent, stub_llm

agent = SimpleAgent(stub_llm, name="researcher")
result = agent.run("Research AI trends", max_steps=3)
print(result.final)
```

### 2. Conversation History (`agents/conversation.py`)

Multi-turn conversation tracking with context management.

**Key Classes:**
- `Message`: Single message with role, content, timestamp
- `ConversationHistory`: Manages conversation with windowing

**Features:**
- Context window limiting (keep last N messages)
- System message pinning
- OpenAI API format compatibility
- Save/load to JSON
- Message role tracking (user, assistant, system, tool)

**Example:**
```python
from agents.conversation import ConversationHistory

conv = ConversationHistory(max_messages=50, system_message="You are helpful")
conv.add_user_message("Hello")
conv.add_assistant_message("Hi! How can I help?")

# Format for LLM API
messages = conv.format_for_openai()

# Save conversation
conv.save("conversation.json")
```

### 3. Tool System (`agents/tools.py`)

Extensible tool/function calling framework.

**Key Classes:**
- `Tool`: Abstract base class for tools
- `ToolRegistry`: Manages available tools
- `ToolResult`: Standardized tool output

**Built-in Tools:**
- `CalculatorTool`: Mathematical calculations
- `WebSearchTool`: Simulated web search
- `TextAnalysisTool`: Text property analysis

**Example:**
```python
from agents.tools import create_default_registry

registry = create_default_registry()

# Execute tool
result = registry.execute("calculator", expression="sqrt(144) + 10")
print(result.output)  # 22.0

# List available tools
print(registry.list_tools())  # ['calculator', 'web_search', 'text_analysis']
```

**Creating Custom Tools:**
```python
from agents.tools import Tool, ToolResult

class CustomTool(Tool):
    @property
    def name(self) -> str:
        return "my_tool"
    
    @property
    def description(self) -> str:
        return "Does custom processing"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string"}
            }
        }
    
    def execute(self, input: str, **kwargs) -> ToolResult:
        # Custom logic here
        output = input.upper()
        return ToolResult(success=True, output=output)

# Register and use
registry.register(CustomTool())
```

### 4. State Management (`agents/state.py`)

Stateful context tracking for agents.

**Key Classes:**
- `AgentState`: Key-value state store with history
- `SessionManager`: Multi-session state management
- `StateSnapshot`: Point-in-time state capture

**Features:**
- Nested state with dot notation (`user.name`)
- State history with snapshots
- Restore to previous states
- Session isolation
- Persistence to JSON

**Example:**
```python
from agents.state import AgentState, SessionManager

# Basic state
state = AgentState()
state.set("user.name", "Alice")
state.set("task.progress", 0.5)

# Take snapshot
state.take_snapshot("checkpoint_1")

# Update state
state.set("task.progress", 1.0)

# Restore previous state
state.restore_snapshot(-1)

# Session management
manager = SessionManager()
session = manager.create_session("user_123", {"role": "researcher"})
session.set("current_task", "AI research")
```

### 5. Chain Composition (`agents/chains.py`)

Compose multi-step agent workflows.

**Key Classes:**
- `SequentialChain`: Execute steps in sequence
- `ParallelChain`: Execute steps in parallel (simulated)
- `ConditionalChain`: Route based on conditions
- `ChainStep`: Single step in a chain
- `ChainResult`: Chain execution result

**Example:**
```python
from agents.chains import SequentialChain, ChainStep

def research(topic): return f"Research on {topic}"
def summarize(text): return f"Summary: {text[:50]}"

steps = [
    ChainStep(name="research", function=research),
    ChainStep(name="summarize", function=summarize)
]

chain = SequentialChain(steps, name="research_pipeline")
result = chain.run("AI trends")

print(result.outputs)  # ['Research on AI trends', 'Summary: Research on AI trends']
```

### 6. Callbacks (`agents/callbacks.py`)

Monitor and respond to agent lifecycle events.

**Key Classes:**
- `Callback`: Abstract base for callbacks
- `LoggingCallback`: Console logging
- `FileCallback`: JSON file logging
- `MetricsCallback`: Performance metrics
- `CallbackManager`: Manages multiple callbacks

**Events:**
- `on_agent_start`: Agent begins execution
- `on_agent_complete`: Agent finishes successfully
- `on_agent_error`: Agent encounters error
- `on_tool_use`: Agent uses a tool
- `on_chain_step`: Chain step completes

**Example:**
```python
from agents.callbacks import CallbackManager, LoggingCallback, MetricsCallback

# Create callbacks
logging = LoggingCallback(verbose=True)
metrics = MetricsCallback()

manager = CallbackManager([logging, metrics])

# In your agent code
manager.on_agent_start("researcher", {"query": "AI"})
# ... agent execution ...
manager.on_agent_complete("researcher", "results", 0.5)

# Get metrics
summary = metrics.get_summary()
print(f"Success rate: {summary['success_rate']}")
```

### 7. Evaluation (`agents/evaluation.py`)

Benchmark and evaluate agent performance.

**Key Classes:**
- `AgentEvaluator`: Evaluate single agent
- `BenchmarkSuite`: Test case collection
- `ComparativeEvaluator`: Compare multiple agents
- `EvaluationResult`: Single evaluation result

**Metrics:**
- Success rate
- Execution time (mean, median, p95, p99)
- Error rate
- Custom scoring

**Example:**
```python
from agents.evaluation import AgentEvaluator, BenchmarkSuite

# Create test suite
suite = BenchmarkSuite(name="text_processing", test_cases=[])
suite.add_test_case("task1", input_data="hello", expected_output="HELLO")
suite.add_test_case("task2", input_data="world", expected_output="WORLD")

# Evaluate agent
def my_agent(input_text):
    return input_text.upper()

evaluator = AgentEvaluator("my_agent")
evaluator.evaluate_suite(suite, my_agent)

# Get metrics
metrics = evaluator.get_metrics()
print(f"Success rate: {metrics['success_rate']:.2%}")
print(f"Mean time: {metrics['execution_time']['mean']:.4f}s")
```

## Architecture Patterns

### Pattern 1: Research → Write → Review

Multi-agent collaboration with specialized roles.

```python
from agents.multi_agent_demo import MultiAgentOrchestrator

orchestrator = MultiAgentOrchestrator()
result = orchestrator.research_write_review(
    topic="AI applications",
    style="informative"
)
```

### Pattern 2: Tool-Using Agent

Agent with external tool capabilities.

```python
from agents.tools import create_default_registry
from agents.multi_agent_demo import ResearchAgent

registry = create_default_registry()
researcher = ResearchAgent(registry)

# Agent uses tools automatically
findings = researcher.research("quantum computing")
```

### Pattern 3: Stateful Workflow

Agent maintains state across multiple steps.

```python
from agents.state import AgentState

state = AgentState({"stage": "planning"})

# Step 1
state.set("stage", "research")
state.set("sources", [...])
state.take_snapshot("after_research")

# Step 2
state.set("stage", "writing")
state.set("draft", "...")

# Restore if needed
state.restore_snapshot(-1)
```

### Pattern 4: Monitored Execution

Track agent execution with callbacks.

```python
from agents.callbacks import CallbackManager, FileCallback, MetricsCallback

callbacks = CallbackManager([
    FileCallback("outputs/agent_log.json"),
    MetricsCallback()
])

# Integrate into agent execution
callbacks.on_agent_start("my_agent", input_data)
result = agent.run(input_data)
callbacks.on_agent_complete("my_agent", result, execution_time)
```

## Usage Examples

### Complete Example: Research Agent with Tools and Callbacks

```python
from agents.tools import create_default_registry
from agents.conversation import ConversationHistory
from agents.callbacks import CallbackManager, LoggingCallback, MetricsCallback
from agents.state import AgentState

# Setup
registry = create_default_registry()
conversation = ConversationHistory(system_message="You are a researcher")
state = AgentState({"topic": "AI", "stage": "planning"})
callbacks = CallbackManager([LoggingCallback(), MetricsCallback()])

# Agent execution
callbacks.on_agent_start("researcher", {"topic": "AI trends"})

# Use tools
calc_result = registry.execute("calculator", expression="2 + 2")
callbacks.on_tool_use("researcher", "calculator", {"expression": "2+2"}, calc_result.output)

# Update conversation
conversation.add_user_message("Research AI trends")
conversation.add_assistant_message("Research findings...")

# Update state
state.set("stage", "research_complete")
state.set("findings", "...")

callbacks.on_agent_complete("researcher", state.get_all(), 1.5)

# Get metrics
metrics = callbacks.callbacks[1].get_summary()
print(metrics)
```

## Best Practices

### 1. Use Context Windows
Keep conversation history manageable:
```python
conv = ConversationHistory(max_messages=20)  # Prevent token overflow
```

### 2. Take State Snapshots
Checkpoint important state changes:
```python
state.take_snapshot("before_critical_operation")
# ... operation ...
if error:
    state.restore_snapshot(-1)
```

### 3. Monitor with Callbacks
Always use callbacks in production:
```python
callbacks = CallbackManager([
    FileCallback("logs/agent.json"),  # Persistent logs
    MetricsCallback()  # Performance tracking
])
```

### 4. Evaluate Regularly
Benchmark agent performance:
```python
evaluator = AgentEvaluator("my_agent")
evaluator.evaluate_suite(test_suite, agent_fn)
metrics = evaluator.get_metrics()
```

### 5. Use Chains for Complex Workflows
Compose agents with chains:
```python
chain = SequentialChain([
    ChainStep("research", research_fn),
    ChainStep("analyze", analyze_fn),
    ChainStep("summarize", summarize_fn)
])
```

### 6. Isolate Sessions
Use session management for multi-user scenarios:
```python
manager = SessionManager()
session = manager.create_session(user_id)
# Isolated state per user
```

## Integration with LLM APIs

### OpenAI Integration Example

```python
import openai
from agents.conversation import ConversationHistory

conv = ConversationHistory(system_message="You are helpful")
conv.add_user_message("Hello")

# Use OpenAI API
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=conv.format_for_openai()
)

conv.add_assistant_message(response.choices[0].message.content)
```

### Tool Calling with OpenAI

```python
from agents.tools import create_default_registry

registry = create_default_registry()
functions = registry.to_openai_functions()

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[...],
    functions=functions,
    function_call="auto"
)

# If function call requested
if response.choices[0].message.get("function_call"):
    tool_name = response.choices[0].message.function_call.name
    tool_args = json.loads(response.choices[0].message.function_call.arguments)
    result = registry.execute(tool_name, **tool_args)
```

## Extending the Framework

### Adding Custom Agents

Extend `SpecializedAgent` from `multi_agent_demo.py`:

```python
from agents.multi_agent_demo import SpecializedAgent

class AnalystAgent(SpecializedAgent):
    def __init__(self, tool_registry=None):
        super().__init__(
            role="analyst",
            system_prompt="You analyze data and provide insights",
            tool_registry=tool_registry
        )
    
    def analyze(self, data):
        # Custom analysis logic
        return self.process(f"Analyze: {data}")
```

### Adding Custom Callbacks

Extend `Callback` base class:

```python
from agents.callbacks import Callback

class SlackCallback(Callback):
    def on_agent_error(self, agent_name, error):
        # Send Slack notification
        send_slack_alert(f"Agent {agent_name} failed: {error}")
```

## Performance Considerations

1. **Context Window Management**: Limit conversation history to prevent token overflow
2. **Tool Execution**: Tools run synchronously; consider async for I/O-bound tools
3. **State Snapshots**: Disable history tracking if not needed: `state.disable_history()`
4. **Parallel Chains**: Currently simulated; implement with asyncio for true parallelism
5. **Callback Overhead**: Limit callbacks in production; use sampling if needed

## Troubleshooting

**Issue**: Agent not using tools
- Ensure `ToolRegistry` is passed to agent initialization
- Check tool is registered: `registry.list_tools()`

**Issue**: Conversation context too large
- Reduce `max_messages` in `ConversationHistory`
- Clear old messages: `conv.clear(keep_system=True)`

**Issue**: State not persisting
- Call `state.save(path)` explicitly
- Verify write permissions on output directory

**Issue**: Callbacks not firing
- Ensure `CallbackManager` is used in agent code
- Check callback methods are implemented correctly

## Further Reading

- `examples/`: Working examples and demos
- `tests/`: Test files demonstrate API usage
- `scripts/`: Utility scripts for running agents
- `docs/`: Additional documentation
