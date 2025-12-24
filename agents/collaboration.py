from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import time
from queue import Queue


class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Message:
    sender: str
    recipient: str
    content: Any
    message_type: str = "default"
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reply_to: Optional[str] = None


@dataclass
class AgentState:
    agent_id: str
    status: str = "idle"
    current_task: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MessageBroker:
    """Central message broker for agent communication."""

    def __init__(self):
        self.queues: Dict[str, Queue] = {}
        self.message_history: List[Message] = []
        self.agents: Dict[str, AgentState] = {}

    def register_agent(self, agent_id: str, capabilities: Optional[List[str]] = None) -> None:
        """Register an agent with the broker."""
        if agent_id not in self.queues:
            self.queues[agent_id] = Queue()
        self.agents[agent_id] = AgentState(
            agent_id=agent_id,
            capabilities=capabilities or [],
        )

    def send_message(self, message: Message) -> bool:
        """Send a message to an agent."""
        if message.recipient not in self.queues:
            return False

        self.queues[message.recipient].put((message.priority.value, message))
        self.message_history.append(message)
        return True

    def receive_message(self, agent_id: str, timeout: float = 0.1) -> Optional[Message]:
        """Receive a message for an agent."""
        if agent_id not in self.queues:
            return None

        try:
            _, message = self.queues[agent_id].get(timeout=timeout)
            return message
        except:
            return None

    def broadcast(self, sender: str, content: Any, message_type: str = "broadcast") -> int:
        """Broadcast message to all agents except sender."""
        count = 0
        for agent_id in self.agents.keys():
            if agent_id != sender:
                msg = Message(
                    sender=sender,
                    recipient=agent_id,
                    content=content,
                    message_type=message_type,
                )
                if self.send_message(msg):
                    count += 1
        return count

    def update_agent_status(self, agent_id: str, status: str, task: Optional[str] = None) -> None:
        """Update agent status."""
        if agent_id in self.agents:
            self.agents[agent_id].status = status
            self.agents[agent_id].current_task = task

    def get_message_stats(self) -> Dict[str, Any]:
        """Get messaging statistics."""
        return {
            "total_messages": len(self.message_history),
            "registered_agents": len(self.agents),
            "active_agents": sum(1 for a in self.agents.values() if a.status == "active"),
        }


class CollaborativeAgent:
    """Agent capable of collaboration through message passing."""

    def __init__(
        self,
        agent_id: str,
        broker: MessageBroker,
        capabilities: Optional[List[str]] = None,
    ):
        self.agent_id = agent_id
        self.broker = broker
        self.capabilities = capabilities or []
        self.message_handlers: Dict[str, Callable] = {}
        self.broker.register_agent(agent_id, capabilities)

    def register_handler(self, message_type: str, handler: Callable[[Message], Any]) -> None:
        """Register a handler for a message type."""
        self.message_handlers[message_type] = handler

    def send(self, recipient: str, content: Any, message_type: str = "default") -> bool:
        """Send a message to another agent."""
        msg = Message(
            sender=self.agent_id,
            recipient=recipient,
            content=content,
            message_type=message_type,
        )
        return self.broker.send_message(msg)

    def broadcast(self, content: Any, message_type: str = "broadcast") -> int:
        """Broadcast to all other agents."""
        return self.broker.broadcast(self.agent_id, content, message_type)

    def process_messages(self, max_messages: int = 10) -> List[Any]:
        """Process incoming messages."""
        results = []
        for _ in range(max_messages):
            msg = self.broker.receive_message(self.agent_id)
            if not msg:
                break

            handler = self.message_handlers.get(msg.message_type)
            if handler:
                result = handler(msg)
                results.append(result)

        return results

    def update_status(self, status: str, task: Optional[str] = None) -> None:
        """Update agent status in broker."""
        self.broker.update_agent_status(self.agent_id, status, task)


class TaskCoordinator:
    """Coordinates task distribution among agents."""

    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self.task_queue: List[Dict[str, Any]] = []

    def assign_task(self, task_id: str, task_description: str, required_capability: Optional[str] = None) -> Optional[str]:
        """Assign task to capable agent."""
        candidates = []

        for agent_id, agent_state in self.broker.agents.items():
            if agent_state.status == "idle":
                if not required_capability or required_capability in agent_state.capabilities:
                    candidates.append(agent_id)

        if not candidates:
            self.task_queue.append({
                "task_id": task_id,
                "description": task_description,
                "capability": required_capability,
            })
            return None

        # Assign to first available agent
        selected = candidates[0]
        self.task_assignments[task_id] = selected

        msg = Message(
            sender="coordinator",
            recipient=selected,
            content={"task_id": task_id, "description": task_description},
            message_type="task_assignment",
            priority=MessagePriority.HIGH,
        )
        self.broker.send_message(msg)
        self.broker.update_agent_status(selected, "active", task_id)

        return selected

    def complete_task(self, task_id: str) -> None:
        """Mark task as complete."""
        if task_id in self.task_assignments:
            agent_id = self.task_assignments[task_id]
            self.broker.update_agent_status(agent_id, "idle")
            del self.task_assignments[task_id]

    def get_task_status(self) -> Dict[str, Any]:
        """Get task coordination status."""
        return {
            "active_tasks": len(self.task_assignments),
            "queued_tasks": len(self.task_queue),
            "assignments": self.task_assignments.copy(),
        }


class ConsensusProtocol:
    """Simple consensus mechanism for multi-agent decisions."""

    def __init__(self, broker: MessageBroker, threshold: float = 0.5):
        self.broker = broker
        self.threshold = threshold
        self.votes: Dict[str, Dict[str, Any]] = {}

    def propose(self, proposal_id: str, proposal: str, proposer: str) -> None:
        """Propose a decision to all agents."""
        self.votes[proposal_id] = {
            "proposal": proposal,
            "proposer": proposer,
            "votes": {},
            "timestamp": time.time(),
        }

        self.broker.broadcast(
            proposer,
            {"proposal_id": proposal_id, "proposal": proposal},
            message_type="consensus_proposal",
        )

    def vote(self, proposal_id: str, agent_id: str, vote: bool) -> None:
        """Record a vote on a proposal."""
        if proposal_id in self.votes:
            self.votes[proposal_id]["votes"][agent_id] = vote

    def tally(self, proposal_id: str) -> Optional[bool]:
        """Tally votes and determine consensus."""
        if proposal_id not in self.votes:
            return None

        votes = self.votes[proposal_id]["votes"]
        if not votes:
            return None

        yes_votes = sum(1 for v in votes.values() if v)
        total_votes = len(votes)

        return (yes_votes / total_votes) >= self.threshold

    def get_consensus_summary(self, proposal_id: str) -> str:
        """Get summary of consensus status."""
        if proposal_id not in self.votes:
            return "Proposal not found"

        data = self.votes[proposal_id]
        yes_count = sum(1 for v in data["votes"].values() if v)
        total = len(data["votes"])
        result = self.tally(proposal_id)

        return (
            f"Proposal: {data['proposal']}\n"
            f"Votes: {yes_count}/{total} ({yes_count/total*100:.0f}% yes)\n"
            f"Consensus: {'REACHED' if result else 'NOT REACHED'}"
        )
