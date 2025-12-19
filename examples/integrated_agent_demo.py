"""Integrated agent demo combining tools, memory, and reasoning chains."""

from agents.tool_registry import create_default_tools
from agents.memory_enhanced import EnhancedMemory
from agents.reasoning_chains import ChainOfThought, SelfReflection, TreeOfThoughts


def demo_agent_with_tools_and_memory():
    """Demonstrate an agent using tools and memory together."""
    print("\n=== Integrated Agent Demo: Tools + Memory + Reasoning ===\n")

    # Setup
    tools = create_default_tools()
    memory = EnhancedMemory()
    cot = ChainOfThought()

    # Record initial observation
    cot.observe("User asks: What is 2 + 3 and what's the current time?")
    memory.working.push("Processing multi-part query")

    print(tools.describe_tools())

    # Use tools
    print("\nExecuting actions...\n")

    # Action 1: Calculate
    cot.think("Need to calculate 2 + 3")
    result1 = tools.execute_tool("calculate", expression="2 + 3")
    memory.record_action("calculate", f"Result: {result1.output}", tool_used="calculate")
    print(f"✓ Calculate result: {result1.output}")
    cot.act(f"Called calculate tool, got {result1.output}")

    # Action 2: Get time
    cot.think("Need to get current time")
    result2 = tools.execute_tool("get_time")
    memory.record_action("get_time", f"Time: {result2.output}", tool_used="get_time")
    print(f"✓ Current time: {result2.output}")
    cot.act(f"Called get_time tool, got {result2.output}")

    # Action 3: Summarize
    long_text = "Agentic AI combines planning, tool use, memory, and reasoning. Tools enable agents to interact with external systems. Memory allows learning and context retention."
    cot.think("Should summarize the long text")
    result3 = tools.execute_tool("summarize", text=long_text, max_length=50)
    memory.record_action("summarize", f"Summary: {result3.output}", tool_used="summarize")
    print(f"✓ Summarize result: {result3.output}")
    cot.act("Summarized text successfully")

    # Record entities
    print("\n--- Entity Tracking ---")
    memory.remember_entity("GPT-4", "Model", provider="OpenAI", capabilities=["text", "vision"])
    memory.remember_entity("Claude", "Model", provider="Anthropic", capabilities=["reasoning"])
    print("Recorded models in memory")

    # Reflection
    reflection = SelfReflection(cot)
    reflection.reflect("All tools executed successfully")
    reflection.reflect("Memory tracking enabled accurate action history")

    # Display results
    print("\n--- Reasoning Trace ---")
    print(cot.format_trace())
    print(reflection.format_reflection())
    print(memory.recall_summary())


def demo_tree_of_thoughts_with_memory():
    """Demonstrate Tree of Thoughts exploration with memory."""
    print("\n=== Tree of Thoughts: Planning with Memory ===\n")

    memory = EnhancedMemory()
    tools = create_default_tools()

    # Problem
    print("Problem: How to gather and summarize information?\n")

    tree = TreeOfThoughts("Gather information on AI agents")

    # Branch possibilities
    branch1 = ["Search web for AI agent papers", "Extract key concepts"]
    branch2 = ["Review local knowledge base", "Filter by relevance"]
    branch3 = ["Query existing summaries", "Combine results"]

    tree.branch(branch1, scores=[0.8, 0.75])
    tree.navigate_to(tree.root.children[0])
    tree.branch([
        "Found 'Planning Agents' paper",
        "Found 'Tool Use' paper",
        "Found 'Memory Networks' paper"
    ], scores=[0.9, 0.85, 0.8])

    # Show planning tree
    print("Planning Tree:")
    print(tree.format_tree())

    # Simulate execution of best path
    print("\nExecuting best path:")
    best = tree.select_best_child()
    if best:
        tree.navigate_to(best)
        path = tree.get_path()
        for step in path:
            memory.record_action(f"Explored: {step}", "In progress")
            print(f"  ✓ {step}")

    print(memory.recall_summary())


def demo_collaborative_agents():
    """Demonstrate multiple agents collaborating with shared memory."""
    print("\n=== Collaborative Agents with Shared Memory ===\n")

    memory = EnhancedMemory()
    tools = create_default_tools()

    agents = {
        "Planner": ChainOfThought(),
        "Executor": ChainOfThought(),
        "Evaluator": ChainOfThought(),
    }

    # Planner agent
    print("1. Planner Agent:")
    agents["Planner"].observe("Need to solve: Calculate 15 * 3 and summarize result")
    agents["Planner"].think("Break into subtasks: calc and summarize")
    agents["Planner"].act("Create execution plan")
    memory.record_action("Planning", "Tasks decomposed", tool_used="planner")
    print("   → Planned task execution\n")

    # Executor agent
    print("2. Executor Agent:")
    agents["Executor"].observe("Execute: 15 * 3")
    calc_result = tools.execute_tool("calculate", expression="15 * 3")
    agents["Executor"].act(f"Executed calculation: {calc_result.output}")
    memory.record_action("Execution", f"Result: {calc_result.output}", tool_used="executor")
    print(f"   → Computed result: {calc_result.output}\n")

    # Evaluator agent
    print("3. Evaluator Agent:")
    agents["Evaluator"].observe(f"Review result: {calc_result.output}")
    agents["Evaluator"].think("45 is mathematically correct")
    agents["Evaluator"].act("Validated: 15 * 3 = 45 ✓")
    memory.record_action("Evaluation", "Result validated", tool_used="evaluator")
    print("   → Validated result\n")

    # Show collaboration summary
    print("--- Collaboration Summary ---")
    print(memory.recall_summary())
    print("\nTimeline of collaboration:")
    print(memory.episodic.get_timeline())


def main():
    demo_agent_with_tools_and_memory()
    demo_tree_of_thoughts_with_memory()
    demo_collaborative_agents()


if __name__ == "__main__":
    main()
