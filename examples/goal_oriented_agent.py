"""Goal-oriented agent demo: decompose goal -> plan -> execute stubs -> critique -> refine."""

from agents.goal_decomposition import GoalDecomposer
from agents.tool_registry import create_default_tools
from agents.critic import AgentCritic
from agents.memory_enhanced import EnhancedMemory


def main():
    print("\n=== Goal-Oriented Agent Demo ===\n")

    goal = "Research retrieval-augmented generation and create a concise summary"
    print(f"Goal: {goal}\n")

    # Decompose goal into tasks
    decomposer = GoalDecomposer()
    plan = decomposer.decompose(goal)
    print("Plan:")
    print(decomposer.visualize_plan(plan))

    # Execution (stubbed with tools)
    tools = create_default_tools()
    memory = EnhancedMemory()

    print("\nExecuting plan stubs:\n")
    for task in decomposer.get_execution_order(plan):
        if "Gather" in task.description:
            res = tools.execute_tool("search", query="retrieval augmented generation overview")
            memory.record_action("search", res.output, tool_used="search")
            print(f"- {task.description}: DONE")
        elif "Process" in task.description:
            long_text = (
                "Retrieval augmented generation (RAG) combines document retrieval with generation to ground responses. "
                "It improves factuality by providing relevant context to the generator."
            )
            res = tools.execute_tool("summarize", text=long_text, max_length=100)
            memory.record_action("summarize", res.output, tool_used="summarize")
            print(f"- {task.description}: DONE")
        elif "Formulate" in task.description:
            print(f"- {task.description}: DONE")

    # Critique the final output (stub)
    final_output = (
        "RAG ties retrieval to generation so answers can cite context. "
        "It improves factuality and relevance by injecting documents into prompts."
    )
    critic = AgentCritic()
    critiques = critic.critique_output(
        final_output,
        expected_criteria={
            "expected_format": "text",
            "required_keywords": ["retrieval", "generation", "context"],
            "min_length": 80,
        },
    )

    print("\nCritique Report:\n")
    print(critic.format_report(critiques))

    if critic.should_retry(critiques):
        print("\nImprovement Prompt:\n")
        print(critic.get_improvement_prompt(critiques))

    print("\nMemory Summary:\n")
    print(memory.recall_summary())


if __name__ == "__main__":
    main()
