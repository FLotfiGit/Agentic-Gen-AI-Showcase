"""Agent state management for tracking context across steps and sessions.

This module provides classes for managing agent state, including key-value stores,
session management, and persistent state storage.
"""
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import copy


@dataclass
class StateSnapshot:
    """A snapshot of agent state at a point in time."""
    timestamp: str
    state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "state": self.state,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateSnapshot':
        return cls(**data)


class AgentState:
    """Manages agent state as a key-value store with history tracking.
    
    Features:
    - Get/set/delete state variables
    - State history with snapshots
    - Reset to previous snapshots
    - Persistence to JSON
    - Nested state support
    """
    
    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        """Initialize agent state.
        
        Args:
            initial_state: Optional initial state dictionary
        """
        self._state: Dict[str, Any] = initial_state or {}
        self._history: List[StateSnapshot] = []
        self._track_history = True
        
        # Take initial snapshot if state provided
        if initial_state:
            self._take_snapshot("initialization")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a state variable.
        
        Args:
            key: State variable key (supports dot notation for nested: 'user.name')
            default: Default value if key not found
            
        Returns:
            State value or default
        """
        # Support nested keys with dot notation
        if '.' in key:
            keys = key.split('.')
            value = self._state
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                    if value is None:
                        return default
                else:
                    return default
            return value
        
        return self._state.get(key, default)
    
    def set(self, key: str, value: Any, take_snapshot: bool = False) -> None:
        """Set a state variable.
        
        Args:
            key: State variable key (supports dot notation for nested)
            value: Value to set
            take_snapshot: Whether to take a snapshot after setting
        """
        # Support nested keys with dot notation
        if '.' in key:
            keys = key.split('.')
            current = self._state
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            self._state[key] = value
        
        if take_snapshot:
            self._take_snapshot(f"set_{key}")
    
    def delete(self, key: str) -> None:
        """Delete a state variable.
        
        Args:
            key: State variable key
        """
        if '.' in key:
            keys = key.split('.')
            current = self._state
            for k in keys[:-1]:
                if k not in current:
                    return
                current = current[k]
            current.pop(keys[-1], None)
        else:
            self._state.pop(key, None)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple state variables at once.
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def get_all(self) -> Dict[str, Any]:
        """Get a copy of the entire state.
        
        Returns:
            Copy of state dictionary
        """
        return copy.deepcopy(self._state)
    
    def clear(self) -> None:
        """Clear all state."""
        self._state = {}
        if self._track_history:
            self._take_snapshot("clear")
    
    def _take_snapshot(self, metadata_label: str = "") -> None:
        """Take a snapshot of current state.
        
        Args:
            metadata_label: Optional label for the snapshot
        """
        snapshot = StateSnapshot(
            timestamp=datetime.now().isoformat(),
            state=copy.deepcopy(self._state),
            metadata={"label": metadata_label}
        )
        self._history.append(snapshot)
    
    def take_snapshot(self, label: str = "") -> None:
        """Manually take a snapshot of current state.
        
        Args:
            label: Label for the snapshot
        """
        if self._track_history:
            self._take_snapshot(label)
    
    def get_history(self) -> List[StateSnapshot]:
        """Get state history.
        
        Returns:
            List of state snapshots
        """
        return self._history.copy()
    
    def restore_snapshot(self, index: int = -1) -> None:
        """Restore state from a snapshot.
        
        Args:
            index: Snapshot index (default -1 for most recent)
        """
        if not self._history:
            return
        
        snapshot = self._history[index]
        self._state = copy.deepcopy(snapshot.state)
    
    def enable_history(self) -> None:
        """Enable state history tracking."""
        self._track_history = True
    
    def disable_history(self) -> None:
        """Disable state history tracking."""
        self._track_history = False
    
    def save(self, path: str) -> None:
        """Save state to JSON file.
        
        Args:
            path: File path to save to
        """
        data = {
            "state": self._state,
            "history": [s.to_dict() for s in self._history]
        }
        Path(path).write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, path: str) -> 'AgentState':
        """Load state from JSON file.
        
        Args:
            path: File path to load from
            
        Returns:
            AgentState instance
        """
        data = json.loads(Path(path).read_text())
        state = cls(initial_state=data.get("state", {}))
        state._history = [StateSnapshot.from_dict(s) for s in data.get("history", [])]
        return state
    
    def __repr__(self) -> str:
        return f"AgentState(keys={list(self._state.keys())}, history={len(self._history)})"


class SessionManager:
    """Manages multiple agent sessions with isolated state.
    
    Each session has its own state and can be saved/loaded independently.
    Useful for multi-user scenarios or parallel agent runs.
    """
    
    def __init__(self):
        """Initialize session manager."""
        self.sessions: Dict[str, AgentState] = {}
        self.active_session: Optional[str] = None
    
    def create_session(self, session_id: str, initial_state: Optional[Dict[str, Any]] = None) -> AgentState:
        """Create a new session.
        
        Args:
            session_id: Unique session identifier
            initial_state: Optional initial state for the session
            
        Returns:
            AgentState for the new session
        """
        state = AgentState(initial_state)
        self.sessions[session_id] = state
        return state
    
    def get_session(self, session_id: str) -> Optional[AgentState]:
        """Get a session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            AgentState or None if not found
        """
        return self.sessions.get(session_id)
    
    def set_active_session(self, session_id: str) -> None:
        """Set the active session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            self.active_session = session_id
    
    def get_active_session(self) -> Optional[AgentState]:
        """Get the active session state.
        
        Returns:
            AgentState of active session or None
        """
        if self.active_session:
            return self.sessions.get(self.active_session)
        return None
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session.
        
        Args:
            session_id: Session identifier
        """
        self.sessions.pop(session_id, None)
        if self.active_session == session_id:
            self.active_session = None
    
    def list_sessions(self) -> List[str]:
        """List all session IDs.
        
        Returns:
            List of session identifiers
        """
        return list(self.sessions.keys())
    
    def save_session(self, session_id: str, path: str) -> None:
        """Save a session to file.
        
        Args:
            session_id: Session identifier
            path: File path to save to
        """
        session = self.get_session(session_id)
        if session:
            session.save(path)
    
    def load_session(self, session_id: str, path: str) -> AgentState:
        """Load a session from file.
        
        Args:
            session_id: Session identifier
            path: File path to load from
            
        Returns:
            Loaded AgentState
        """
        state = AgentState.load(path)
        self.sessions[session_id] = state
        return state
    
    def __repr__(self) -> str:
        return f"SessionManager(sessions={len(self.sessions)}, active={self.active_session})"


def demo_state_management():
    """Demonstrate state management capabilities."""
    print("=" * 70)
    print("Agent State Management Demo")
    print("=" * 70)
    
    # Basic state operations
    print("\n1. Basic State Operations")
    print("-" * 70)
    
    state = AgentState()
    state.set("user", "Alice")
    state.set("task", "research")
    state.set("progress", 0.5)
    
    print(f"User: {state.get('user')}")
    print(f"Task: {state.get('task')}")
    print(f"Progress: {state.get('progress')}")
    print(f"All state: {state.get_all()}")
    
    # Nested state
    print("\n2. Nested State (dot notation)")
    print("-" * 70)
    
    state.set("user.name", "Alice")
    state.set("user.email", "alice@example.com")
    state.set("settings.theme", "dark")
    state.set("settings.language", "en")
    
    print(f"User name: {state.get('user.name')}")
    print(f"Settings theme: {state.get('settings.theme')}")
    print(f"Full state: {json.dumps(state.get_all(), indent=2)}")
    
    # State history
    print("\n3. State History and Snapshots")
    print("-" * 70)
    
    state.take_snapshot("before_updates")
    state.set("progress", 0.75)
    state.take_snapshot("mid_progress")
    state.set("progress", 1.0)
    state.take_snapshot("completed")
    
    print(f"Current progress: {state.get('progress')}")
    print(f"History snapshots: {len(state.get_history())}")
    
    # Restore previous state
    state.restore_snapshot(-2)  # Restore to mid_progress
    print(f"After restore: {state.get('progress')}")
    
    # Session management
    print("\n4. Session Management")
    print("-" * 70)
    
    manager = SessionManager()
    
    # Create sessions
    session1 = manager.create_session("user_123", {"user": "Alice", "role": "researcher"})
    session2 = manager.create_session("user_456", {"user": "Bob", "role": "writer"})
    
    session1.set("progress", 0.8)
    session2.set("progress", 0.3)
    
    print(f"Sessions: {manager.list_sessions()}")
    print(f"Session 1 progress: {session1.get('progress')}")
    print(f"Session 2 progress: {session2.get('progress')}")
    
    # Active session
    manager.set_active_session("user_123")
    active = manager.get_active_session()
    print(f"Active session user: {active.get('user')}")
    
    # Persistence
    print("\n5. State Persistence")
    print("-" * 70)
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    state_path = "outputs/agent_state.json"
    state.save(state_path)
    print(f"State saved to: {state_path}")
    
    # Load state
    loaded_state = AgentState.load(state_path)
    print(f"Loaded state: {loaded_state.get_all()}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_state_management()
