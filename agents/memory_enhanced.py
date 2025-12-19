from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
import time


@dataclass
class Memory:
    content: str
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    importance: float = 1.0

    def age(self) -> float:
        """Get memory age in seconds."""
        return time.time() - self.timestamp


@dataclass
class Entity:
    name: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    first_seen: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    mention_count: int = 0


class SemanticMemory:
    """Semantic memory for facts and knowledge."""

    def __init__(self):
        self.facts: Dict[str, Memory] = {}
        self.entities: Dict[str, Entity] = {}

    def store_fact(
        self,
        key: str,
        content: str,
        tags: Optional[Set[str]] = None,
        importance: float = 1.0,
    ) -> None:
        """Store a semantic fact."""
        self.facts[key] = Memory(
            content=content,
            tags=tags or set(),
            importance=importance,
        )

    def retrieve_fact(self, key: str) -> Optional[str]:
        """Retrieve a fact by key."""
        mem = self.facts.get(key)
        return mem.content if mem else None

    def search_by_tag(self, tag: str) -> List[Memory]:
        """Search facts by tag."""
        return [mem for mem in self.facts.values() if tag in mem.tags]

    def add_entity(self, name: str, entity_type: str, **props) -> Entity:
        """Track an entity."""
        if name in self.entities:
            entity = self.entities[name]
            entity.last_updated = time.time()
            entity.mention_count += 1
            entity.properties.update(props)
        else:
            entity = Entity(name=name, entity_type=entity_type, properties=props)
            self.entities[name] = entity
        return entity

    def get_entity(self, name: str) -> Optional[Entity]:
        """Retrieve entity information."""
        return self.entities.get(name)

    def list_entities(self, entity_type: Optional[str] = None) -> List[Entity]:
        """List entities, optionally filtered by type."""
        entities = list(self.entities.values())
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]
        return sorted(entities, key=lambda e: e.mention_count, reverse=True)


class EpisodicMemory:
    """Episodic memory for event sequences."""

    def __init__(self, max_events: int = 1000):
        self.events: List[Memory] = []
        self.max_events = max_events
        self.event_index: Dict[str, List[int]] = {}

    def record_event(
        self,
        description: str,
        event_type: str = "action",
        **context,
    ) -> None:
        """Record an event with context."""
        mem = Memory(content=description, tags={event_type}, context=context)
        self.events.append(mem)

        if event_type not in self.event_index:
            self.event_index[event_type] = []
        self.event_index[event_type].append(len(self.events) - 1)

        if len(self.events) > self.max_events:
            self.events.pop(0)

    def get_recent_events(self, count: int = 10, event_type: Optional[str] = None) -> List[Memory]:
        """Get recent events."""
        events = self.events[-count:]
        if event_type:
            events = [e for e in events if event_type in e.tags]
        return events

    def get_events_by_type(self, event_type: str) -> List[Memory]:
        """Get all events of a specific type."""
        indices = self.event_index.get(event_type, [])
        return [self.events[i] for i in indices if i < len(self.events)]

    def get_timeline(self, start_idx: int = 0, end_idx: Optional[int] = None) -> str:
        """Get event timeline as text."""
        events = self.events[start_idx:end_idx]
        lines = ["Event Timeline:\n"]
        for i, event in enumerate(events, 1):
            age = event.age()
            age_str = f"{age:.0f}s ago" if age < 3600 else f"{age / 3600:.1f}h ago"
            lines.append(f"{i}. [{age_str}] {event.content}")
        return "\n".join(lines)


class WorkingMemory:
    """Working memory for current context."""

    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self.items: List[Memory] = []

    def push(self, content: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Add to working memory."""
        mem = Memory(content=content, context=context or {})
        self.items.append(mem)
        if len(self.items) > self.capacity:
            self.items.pop(0)

    def get_context(self) -> str:
        """Get current working context."""
        lines = ["Current Context:\n"]
        for i, item in enumerate(self.items, 1):
            lines.append(f"{i}. {item.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear working memory."""
        self.items.clear()


class EnhancedMemory:
    """Unified memory system combining semantic, episodic, and working memory."""

    def __init__(self):
        self.semantic = SemanticMemory()
        self.episodic = EpisodicMemory()
        self.working = WorkingMemory()

    def record_action(
        self,
        action: str,
        result: str,
        tool_used: Optional[str] = None,
    ) -> None:
        """Record an agent action."""
        self.episodic.record_event(
            f"Action: {action} → Result: {result}",
            event_type="action",
            tool=tool_used,
        )
        self.working.push(f"Just executed: {action}")

    def remember_entity(self, name: str, entity_type: str, **properties) -> None:
        """Track and remember an entity."""
        self.semantic.add_entity(name, entity_type, **properties)
        self.working.push(f"Noted entity: {name} ({entity_type})")

    def recall_summary(self) -> str:
        """Get a summary of memory state."""
        lines = ["\n=== Memory Summary ===\n"]
        lines.append(f"Semantic facts: {len(self.semantic.facts)}")
        lines.append(f"Known entities: {len(self.semantic.entities)}")
        lines.append(f"Event history: {len(self.episodic.events)} events")
        lines.append(f"Working items: {len(self.working.items)}")

        if self.semantic.entities:
            lines.append("\nKey Entities:")
            for entity in self.semantic.list_entities()[:5]:
                lines.append(f"  • {entity.name} ({entity.entity_type}): {entity.mention_count} mentions")

        return "\n".join(lines)
