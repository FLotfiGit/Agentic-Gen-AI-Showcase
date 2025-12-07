"""Multi-agent orchestration demo.

This demo shows multiple specialized agents collaborating:
- Researcher: gathers information using tools
- Writer: creates content based on research
- Critic: reviews and provides feedback

Uses the conversation history and tool calling framework.
"""
from pathlib import Path
from typing import List, Dict, Any
from agents.conversation import ConversationHistory, Message
from agents.tools import ToolRegistry, create_default_registry, ToolResult
from agents.agent_utils import stub_llm


class SpecializedAgent:
    """Base class for specialized agents in a multi-agent system."""
    
    def __init__(self, role: str, system_prompt: str, tool_registry: ToolRegistry = None):
        """Initialize a specialized agent.
        
        Args:
            role: Agent role/name (e.g., 'researcher', 'writer', 'critic')
            system_prompt: System prompt defining agent's behavior
            tool_registry: Optional tool registry for tool-using agents
        """
        self.role = role
        self.system_prompt = system_prompt
        self.tool_registry = tool_registry
        self.conversation = ConversationHistory(
            max_messages=20,
            system_message=system_prompt
        )
    
    def process(self, task: str, context: Dict[str, Any] = None) -> str:
        """Process a task and return the agent's response.
        
        Args:
            task: Task description
            context: Optional context from other agents
            
        Returns:
            Agent's response
        """
        self.conversation.add_user_message(task)
        
        # Build prompt with context
        prompt_parts = [f"[Role: {self.role}]", f"Task: {task}"]
        if context:
            prompt_parts.append(f"Context: {context}")
        
        # Use stub LLM (in production, use actual LLM)
        response = stub_llm("\n".join(prompt_parts))
        
        self.conversation.add_assistant_message(response)
        return response
    
    def use_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Use a tool from the registry.
        
        Args:
            tool_name: Name of tool to use
            **kwargs: Tool parameters
            
        Returns:
            ToolResult
        """
        if not self.tool_registry:
            return ToolResult(success=False, output=None, error="No tool registry available")
        
        result = self.tool_registry.execute(tool_name, **kwargs)
        self.conversation.add_tool_result(tool_name, str(result.output))
        return result


class ResearchAgent(SpecializedAgent):
    """Agent specialized in research and information gathering."""
    
    def __init__(self, tool_registry: ToolRegistry = None):
        system_prompt = (
            "You are a research assistant. Your role is to gather information, "
            "verify facts, and use available tools to find answers. "
            "Be thorough and cite sources when possible."
        )
        super().__init__("researcher", system_prompt, tool_registry)
    
    def research(self, topic: str) -> Dict[str, Any]:
        """Research a topic using available tools.
        
        Args:
            topic: Topic to research
            
        Returns:
            Research results dictionary
        """
        findings = {
            "topic": topic,
            "sources": [],
            "summary": ""
        }
        
        # Use web search if available
        if self.tool_registry and self.tool_registry.get_tool("web_search"):
            search_result = self.use_tool("web_search", query=topic, num_results=3)
            if search_result.success:
                findings["sources"] = search_result.output
        
        # Generate summary
        summary_prompt = f"Summarize research findings about: {topic}"
        findings["summary"] = self.process(summary_prompt, {"sources": findings["sources"]})
        
        return findings


class WriterAgent(SpecializedAgent):
    """Agent specialized in content creation and writing."""
    
    def __init__(self):
        system_prompt = (
            "You are a creative writer. Your role is to create engaging, "
            "well-structured content based on research and requirements. "
            "Focus on clarity, coherence, and audience engagement."
        )
        super().__init__("writer", system_prompt)
    
    def write(self, topic: str, research: Dict[str, Any] = None, style: str = "informative") -> str:
        """Write content on a topic.
        
        Args:
            topic: Topic to write about
            research: Optional research findings
            style: Writing style (informative, creative, technical, etc.)
            
        Returns:
            Written content
        """
        context = {
            "style": style,
            "research": research or {}
        }
        
        task = f"Write {style} content about: {topic}"
        return self.process(task, context)


class CriticAgent(SpecializedAgent):
    """Agent specialized in reviewing and providing constructive feedback."""
    
    def __init__(self):
        system_prompt = (
            "You are a constructive critic. Your role is to review content, "
            "identify strengths and weaknesses, and provide specific, "
            "actionable feedback for improvement."
        )
        super().__init__("critic", system_prompt)
    
    def review(self, content: str, criteria: List[str] = None) -> Dict[str, Any]:
        """Review content against criteria.
        
        Args:
            content: Content to review
            criteria: Optional list of criteria to evaluate
            
        Returns:
            Review results with strengths, weaknesses, and suggestions
        """
        if criteria is None:
            criteria = ["clarity", "accuracy", "engagement", "structure"]
        
        task = f"Review this content against criteria: {', '.join(criteria)}\n\nContent: {content}"
        feedback = self.process(task)
        
        return {
            "feedback": feedback,
            "criteria_evaluated": criteria,
            "overall_assessment": "Constructive feedback provided"
        }


class MultiAgentOrchestrator:
    """Orchestrates collaboration between multiple specialized agents."""
    
    def __init__(self, tool_registry: ToolRegistry = None):
        """Initialize the orchestrator with specialized agents.
        
        Args:
            tool_registry: Optional tool registry for agents
        """
        self.tool_registry = tool_registry or create_default_registry()
        self.researcher = ResearchAgent(self.tool_registry)
        self.writer = WriterAgent()
        self.critic = CriticAgent()
        self.workflow_history = []
    
    def research_write_review(self, topic: str, style: str = "informative") -> Dict[str, Any]:
        """Execute a full research → write → review workflow.
        
        Args:
            topic: Topic to process
            style: Writing style
            
        Returns:
            Complete workflow results
        """
        print(f"\n🔬 Starting multi-agent workflow for topic: '{topic}'")
        
        # Step 1: Research
        print("\n[1/3] Researcher gathering information...")
        research = self.researcher.research(topic)
        self.workflow_history.append({"step": "research", "agent": "researcher", "output": research})
        
        # Step 2: Write
        print("\n[2/3] Writer creating content...")
        content = self.writer.write(topic, research, style)
        self.workflow_history.append({"step": "write", "agent": "writer", "output": content})
        
        # Step 3: Review
        print("\n[3/3] Critic reviewing content...")
        review = self.critic.review(content)
        self.workflow_history.append({"step": "review", "agent": "critic", "output": review})
        
        print("\n✅ Multi-agent workflow complete!")
        
        return {
            "topic": topic,
            "research": research,
            "content": content,
            "review": review,
            "workflow_steps": len(self.workflow_history)
        }
    
    def save_workflow(self, path: str) -> None:
        """Save workflow history to file.
        
        Args:
            path: File path to save to
        """
        import json
        Path(path).write_text(json.dumps(self.workflow_history, indent=2))
        print(f"\n💾 Workflow saved to: {path}")


def run_demo():
    """Run a demonstration of multi-agent orchestration."""
    print("=" * 70)
    print("Multi-Agent Orchestration Demo")
    print("=" * 70)
    
    # Create orchestrator with tools
    orchestrator = MultiAgentOrchestrator()
    
    # Example 1: Research and write about AI
    topic1 = "Applications of Large Language Models in Healthcare"
    result1 = orchestrator.research_write_review(topic1, style="informative")
    
    print("\n" + "=" * 70)
    print("WORKFLOW RESULTS")
    print("=" * 70)
    print(f"\nTopic: {result1['topic']}")
    print(f"\nResearch Summary:\n{result1['research']['summary'][:200]}...")
    print(f"\nWritten Content:\n{result1['content'][:200]}...")
    print(f"\nCritic Feedback:\n{result1['review']['feedback'][:200]}...")
    
    # Save workflow
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    orchestrator.save_workflow("outputs/multi_agent_workflow.json")
    
    # Example 2: Calculate something
    print("\n" + "=" * 70)
    print("TOOL USAGE EXAMPLE")
    print("=" * 70)
    
    # Researcher uses calculator
    calc_result = orchestrator.researcher.use_tool(
        "calculator",
        expression="sqrt(144) + 2**3"
    )
    print(f"\nCalculator result: {calc_result}")
    
    # Researcher uses text analysis
    text_result = orchestrator.researcher.use_tool(
        "text_analysis",
        text="This is an amazing and wonderful demonstration of multi-agent collaboration!"
    )
    print(f"\nText analysis result: {text_result.output}")
    
    print("\n" + "=" * 70)
    print(f"Available tools: {orchestrator.tool_registry.list_tools()}")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
