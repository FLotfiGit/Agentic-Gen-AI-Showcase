"""
Example: Reasoning Agent with ReAct Pattern

This example demonstrates how to use the reasoning agent to solve problems
using a combination of reasoning and actions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.reasoning_agent import ReasoningAgent, Tool, get_example_tools


def main():
    print("=" * 60)
    print("REASONING AGENT EXAMPLE - ReAct Pattern")
    print("=" * 60)
    print()
    
    # Example 1: Using built-in tools
    print("Example 1: Math Problem Solving")
    print("-" * 60)
    
    agent = ReasoningAgent(tools=get_example_tools(), max_iterations=5)
    
    result = agent.run(
        task="What is (25 + 15) * 3 - 10?",
        context="Use the calculator tool to solve this step by step."
    )
    
    print(f"Task: {result['task']}")
    print(f"\nExecution Trace:")
    for entry in result['history']:
        entry_type = entry['type']
        if entry_type == 'thought':
            print(f"  💭 Thought: {entry['content']}")
        elif entry_type == 'action':
            print(f"  🔧 Action: {entry['tool']} with {entry['parameters']}")
        elif entry_type == 'observation':
            print(f"  👁️  Observation: {entry['content']}")
        elif entry_type in ['conclusion', 'reflection']:
            print(f"  ✅ {entry_type.capitalize()}: {entry['content']}")
    
    print(f"\nStatus: {result['status']}")
    print(f"Total Steps: {result['steps']}")
    print()
    
    # Example 2: Custom tool
    print("Example 2: Custom Tool - String Manipulation")
    print("-" * 60)
    
    def reverse_text(text: str) -> str:
        """Reverses the input text"""
        return text[::-1]
    
    def count_words(text: str) -> int:
        """Counts words in text"""
        return len(text.split())
    
    custom_tools = [
        Tool(
            name="reverse",
            description="Reverses a string. Input should be the text to reverse.",
            function=reverse_text
        ),
        Tool(
            name="word_counter",
            description="Counts the number of words in text. Input should be the text to analyze.",
            function=count_words
        )
    ]
    
    custom_agent = ReasoningAgent(tools=custom_tools, max_iterations=3)
    
    result2 = custom_agent.run(
        task="Reverse the phrase 'Hello World' and count its words",
        context="Use the available tools to complete this task."
    )
    
    print(f"Task: {result2['task']}")
    print(f"Steps taken: {result2['steps']}")
    print()
    
    # Example 3: Show prompt template
    print("Example 3: ReAct Prompt Template")
    print("-" * 60)
    print("This is the prompt template used with LLMs:")
    print()
    print(agent.get_prompt_template())
    print()
    
    print("=" * 60)
    print("✅ Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
