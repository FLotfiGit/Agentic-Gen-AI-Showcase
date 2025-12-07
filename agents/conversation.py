"""Conversation history management for multi-turn agentic interactions.

This module provides classes to track conversation history with context windowing,
message role tracking, and persistence capabilities.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(**data)


class ConversationHistory:
    """Manages conversation history with context windowing and persistence.
    
    Supports:
    - Adding messages with automatic timestamping
    - Context window limiting (keep last N messages)
    - System message pinning (always included)
    - Save/load to JSON
    - Formatting for different LLM APIs
    """
    
    def __init__(self, max_messages: int = 50, system_message: Optional[str] = None):
        """Initialize conversation history.
        
        Args:
            max_messages: Maximum number of messages to keep (system message not counted)
            system_message: Optional system message that's always included first
        """
        self.max_messages = max_messages
        self.system_message = system_message
        self.messages: List[Message] = []
        
        if system_message:
            self.messages.append(Message(role="system", content=system_message))
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to the conversation history.
        
        Args:
            role: Message role ('user', 'assistant', 'system', 'tool')
            content: Message content
            metadata: Optional metadata dictionary
        """
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        self._apply_context_window()
    
    def add_user_message(self, content: str) -> None:
        """Convenience method to add a user message."""
        self.add_message("user", content)
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Convenience method to add an assistant message."""
        self.add_message("assistant", content, metadata)
    
    def add_tool_result(self, tool_name: str, result: str) -> None:
        """Add a tool execution result."""
        self.add_message("tool", result, metadata={"tool": tool_name})
    
    def _apply_context_window(self) -> None:
        """Trim conversation to max_messages, preserving system message."""
        if len(self.messages) <= self.max_messages + 1:  # +1 for system message
            return
        
        # Keep system message (if exists) and last N messages
        if self.system_message:
            system_msg = self.messages[0]
            self.messages = [system_msg] + self.messages[-(self.max_messages):]
        else:
            self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self, include_system: bool = True) -> List[Message]:
        """Get all messages in the conversation.
        
        Args:
            include_system: Whether to include system message
            
        Returns:
            List of Message objects
        """
        if include_system or not self.system_message:
            return self.messages.copy()
        return [msg for msg in self.messages if msg.role != "system"]
    
    def format_for_openai(self) -> List[Dict[str, str]]:
        """Format messages for OpenAI API (ChatCompletion format).
        
        Returns:
            List of dicts with 'role' and 'content' keys
        """
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]
    
    def format_as_text(self, include_timestamps: bool = False) -> str:
        """Format conversation as readable text.
        
        Args:
            include_timestamps: Whether to include message timestamps
            
        Returns:
            Formatted conversation string
        """
        lines = []
        for msg in self.messages:
            prefix = f"[{msg.timestamp}] " if include_timestamps else ""
            lines.append(f"{prefix}{msg.role.upper()}: {msg.content}")
        return "\n".join(lines)
    
    def save(self, path: str) -> None:
        """Save conversation history to JSON file.
        
        Args:
            path: File path to save to
        """
        data = {
            "max_messages": self.max_messages,
            "system_message": self.system_message,
            "messages": [msg.to_dict() for msg in self.messages]
        }
        Path(path).write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, path: str) -> 'ConversationHistory':
        """Load conversation history from JSON file.
        
        Args:
            path: File path to load from
            
        Returns:
            ConversationHistory instance
        """
        data = json.loads(Path(path).read_text())
        conv = cls(
            max_messages=data.get("max_messages", 50),
            system_message=data.get("system_message")
        )
        # Clear auto-added system message
        conv.messages = []
        # Load all messages
        for msg_data in data.get("messages", []):
            conv.messages.append(Message.from_dict(msg_data))
        return conv
    
    def clear(self, keep_system: bool = True) -> None:
        """Clear conversation history.
        
        Args:
            keep_system: Whether to keep the system message
        """
        if keep_system and self.system_message:
            self.messages = [msg for msg in self.messages if msg.role == "system"]
        else:
            self.messages = []
    
    def get_last_n_messages(self, n: int) -> List[Message]:
        """Get the last N messages.
        
        Args:
            n: Number of messages to retrieve
            
        Returns:
            List of last N messages
        """
        return self.messages[-n:] if n > 0 else []
    
    def __len__(self) -> int:
        """Return number of messages (excluding system message)."""
        system_count = 1 if self.system_message else 0
        return len(self.messages) - system_count
    
    def __repr__(self) -> str:
        return f"ConversationHistory(messages={len(self)}, max={self.max_messages})"
