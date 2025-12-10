"""Callback system for agent lifecycle events.

This module provides a flexible callback system for monitoring and responding
to agent lifecycle events like execution start/complete, errors, and tool usage.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path


@dataclass
class CallbackEvent:
    """An event in the agent lifecycle."""
    event_type: str  # start, complete, error, tool_use, etc.
    timestamp: str
    agent_name: str
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "agent_name": self.agent_name,
            "data": self.data
        }


class Callback(ABC):
    """Base class for agent callbacks.
    
    Implement specific callback methods to respond to agent events.
    """
    
    def on_agent_start(self, agent_name: str, input_data: Any) -> None:
        """Called when an agent starts execution.
        
        Args:
            agent_name: Name of the agent
            input_data: Input provided to the agent
        """
        pass
    
    def on_agent_complete(self, agent_name: str, output: Any, execution_time: float) -> None:
        """Called when an agent completes successfully.
        
        Args:
            agent_name: Name of the agent
            output: Agent's output
            execution_time: Time taken to execute (seconds)
        """
        pass
    
    def on_agent_error(self, agent_name: str, error: Exception) -> None:
        """Called when an agent encounters an error.
        
        Args:
            agent_name: Name of the agent
            error: Exception that occurred
        """
        pass
    
    def on_tool_use(self, agent_name: str, tool_name: str, tool_input: Dict[str, Any], 
                    tool_output: Any) -> None:
        """Called when an agent uses a tool.
        
        Args:
            agent_name: Name of the agent
            tool_name: Name of the tool used
            tool_input: Input provided to the tool
            tool_output: Output from the tool
        """
        pass
    
    def on_chain_step(self, chain_name: str, step_name: str, step_output: Any) -> None:
        """Called after each step in a chain completes.
        
        Args:
            chain_name: Name of the chain
            step_name: Name of the step
            step_output: Output from the step
        """
        pass


class LoggingCallback(Callback):
    """Callback that logs all events to console."""
    
    def __init__(self, verbose: bool = True):
        """Initialize logging callback.
        
        Args:
            verbose: Whether to print detailed information
        """
        self.verbose = verbose
    
    def on_agent_start(self, agent_name: str, input_data: Any) -> None:
        print(f"🚀 Agent '{agent_name}' started")
        if self.verbose:
            print(f"   Input: {str(input_data)[:100]}")
    
    def on_agent_complete(self, agent_name: str, output: Any, execution_time: float) -> None:
        print(f"✅ Agent '{agent_name}' completed in {execution_time:.3f}s")
        if self.verbose:
            print(f"   Output: {str(output)[:100]}")
    
    def on_agent_error(self, agent_name: str, error: Exception) -> None:
        print(f"❌ Agent '{agent_name}' error: {str(error)}")
    
    def on_tool_use(self, agent_name: str, tool_name: str, tool_input: Dict[str, Any], 
                    tool_output: Any) -> None:
        print(f"🔧 Agent '{agent_name}' used tool '{tool_name}'")
        if self.verbose:
            print(f"   Input: {tool_input}")
            print(f"   Output: {str(tool_output)[:100]}")
    
    def on_chain_step(self, chain_name: str, step_name: str, step_output: Any) -> None:
        print(f"⛓️  Chain '{chain_name}' completed step '{step_name}'")


class FileCallback(Callback):
    """Callback that writes events to a JSON file."""
    
    def __init__(self, output_path: str):
        """Initialize file callback.
        
        Args:
            output_path: Path to write events to
        """
        self.output_path = Path(output_path)
        self.events: List[CallbackEvent] = []
        # Create parent directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _add_event(self, event_type: str, agent_name: str, data: Dict[str, Any]) -> None:
        """Add an event to the log.
        
        Args:
            event_type: Type of event
            agent_name: Agent name
            data: Event data
        """
        event = CallbackEvent(
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            data=data
        )
        self.events.append(event)
    
    def on_agent_start(self, agent_name: str, input_data: Any) -> None:
        self._add_event("agent_start", agent_name, {"input": str(input_data)})
    
    def on_agent_complete(self, agent_name: str, output: Any, execution_time: float) -> None:
        self._add_event("agent_complete", agent_name, {
            "output": str(output),
            "execution_time": execution_time
        })
    
    def on_agent_error(self, agent_name: str, error: Exception) -> None:
        self._add_event("agent_error", agent_name, {"error": str(error)})
    
    def on_tool_use(self, agent_name: str, tool_name: str, tool_input: Dict[str, Any], 
                    tool_output: Any) -> None:
        self._add_event("tool_use", agent_name, {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": str(tool_output)
        })
    
    def on_chain_step(self, chain_name: str, step_name: str, step_output: Any) -> None:
        self._add_event("chain_step", chain_name, {
            "step_name": step_name,
            "step_output": str(step_output)
        })
    
    def flush(self) -> None:
        """Write all events to file."""
        events_dict = [event.to_dict() for event in self.events]
        self.output_path.write_text(json.dumps(events_dict, indent=2))
    
    def __del__(self):
        """Flush events when callback is destroyed."""
        if self.events:
            self.flush()


class MetricsCallback(Callback):
    """Callback that collects performance metrics."""
    
    def __init__(self):
        """Initialize metrics callback."""
        self.metrics = {
            "total_agents": 0,
            "successful_agents": 0,
            "failed_agents": 0,
            "total_tools_used": 0,
            "total_execution_time": 0.0,
            "agent_times": {},
            "tool_usage": {}
        }
    
    def on_agent_start(self, agent_name: str, input_data: Any) -> None:
        self.metrics["total_agents"] += 1
    
    def on_agent_complete(self, agent_name: str, output: Any, execution_time: float) -> None:
        self.metrics["successful_agents"] += 1
        self.metrics["total_execution_time"] += execution_time
        
        if agent_name not in self.metrics["agent_times"]:
            self.metrics["agent_times"][agent_name] = []
        self.metrics["agent_times"][agent_name].append(execution_time)
    
    def on_agent_error(self, agent_name: str, error: Exception) -> None:
        self.metrics["failed_agents"] += 1
    
    def on_tool_use(self, agent_name: str, tool_name: str, tool_input: Dict[str, Any], 
                    tool_output: Any) -> None:
        self.metrics["total_tools_used"] += 1
        
        if tool_name not in self.metrics["tool_usage"]:
            self.metrics["tool_usage"][tool_name] = 0
        self.metrics["tool_usage"][tool_name] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary.
        
        Returns:
            Dictionary with aggregated metrics
        """
        success_rate = 0.0
        if self.metrics["total_agents"] > 0:
            success_rate = self.metrics["successful_agents"] / self.metrics["total_agents"]
        
        avg_times = {}
        for agent_name, times in self.metrics["agent_times"].items():
            avg_times[agent_name] = sum(times) / len(times) if times else 0.0
        
        return {
            "total_agents": self.metrics["total_agents"],
            "success_rate": success_rate,
            "total_execution_time": self.metrics["total_execution_time"],
            "average_agent_times": avg_times,
            "total_tools_used": self.metrics["total_tools_used"],
            "tool_usage_breakdown": self.metrics["tool_usage"]
        }


class CallbackManager:
    """Manages multiple callbacks and dispatches events to all registered callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """Initialize callback manager.
        
        Args:
            callbacks: Optional list of initial callbacks
        """
        self.callbacks: List[Callback] = callbacks or []
    
    def add_callback(self, callback: Callback) -> None:
        """Add a callback.
        
        Args:
            callback: Callback instance to add
        """
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callback) -> None:
        """Remove a callback.
        
        Args:
            callback: Callback instance to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def on_agent_start(self, agent_name: str, input_data: Any) -> None:
        """Notify all callbacks of agent start."""
        for callback in self.callbacks:
            try:
                callback.on_agent_start(agent_name, input_data)
            except Exception as e:
                print(f"Callback error in on_agent_start: {e}")
    
    def on_agent_complete(self, agent_name: str, output: Any, execution_time: float) -> None:
        """Notify all callbacks of agent completion."""
        for callback in self.callbacks:
            try:
                callback.on_agent_complete(agent_name, output, execution_time)
            except Exception as e:
                print(f"Callback error in on_agent_complete: {e}")
    
    def on_agent_error(self, agent_name: str, error: Exception) -> None:
        """Notify all callbacks of agent error."""
        for callback in self.callbacks:
            try:
                callback.on_agent_error(agent_name, error)
            except Exception as e:
                print(f"Callback error in on_agent_error: {e}")
    
    def on_tool_use(self, agent_name: str, tool_name: str, tool_input: Dict[str, Any], 
                    tool_output: Any) -> None:
        """Notify all callbacks of tool use."""
        for callback in self.callbacks:
            try:
                callback.on_tool_use(agent_name, tool_name, tool_input, tool_output)
            except Exception as e:
                print(f"Callback error in on_tool_use: {e}")
    
    def on_chain_step(self, chain_name: str, step_name: str, step_output: Any) -> None:
        """Notify all callbacks of chain step completion."""
        for callback in self.callbacks:
            try:
                callback.on_chain_step(chain_name, step_name, step_output)
            except Exception as e:
                print(f"Callback error in on_chain_step: {e}")


def demo_callbacks():
    """Demonstrate callback system."""
    print("=" * 70)
    print("Agent Callbacks Demo")
    print("=" * 70)
    
    # Create callbacks
    logging_cb = LoggingCallback(verbose=True)
    file_cb = FileCallback("outputs/agent_events.json")
    metrics_cb = MetricsCallback()
    
    # Create callback manager
    manager = CallbackManager([logging_cb, file_cb, metrics_cb])
    
    # Simulate agent lifecycle
    print("\nSimulating agent execution with callbacks:")
    print("-" * 70)
    
    import time
    
    # Agent 1
    manager.on_agent_start("researcher", {"query": "AI trends"})
    time.sleep(0.1)
    manager.on_tool_use("researcher", "web_search", {"query": "AI trends"}, ["result1", "result2"])
    manager.on_agent_complete("researcher", "Research complete", 0.15)
    
    # Agent 2
    manager.on_agent_start("writer", {"topic": "AI summary"})
    time.sleep(0.05)
    manager.on_agent_complete("writer", "Article written", 0.08)
    
    # Agent 3 with error
    manager.on_agent_start("validator", {"content": "test"})
    manager.on_agent_error("validator", Exception("Validation failed"))
    
    # Display metrics
    print("\n" + "=" * 70)
    print("Metrics Summary:")
    print("-" * 70)
    summary = metrics_cb.get_summary()
    print(json.dumps(summary, indent=2))
    
    # Flush file callback
    file_cb.flush()
    print(f"\n📄 Events written to: {file_cb.output_path}")
    print("=" * 70)


if __name__ == "__main__":
    demo_callbacks()
