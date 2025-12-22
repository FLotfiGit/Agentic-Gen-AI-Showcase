from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Task:
    description: str
    task_id: str
    parent_id: Optional[str] = None
    subtasks: List[Task] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1

    def add_subtask(self, task: Task) -> None:
        """Add a subtask."""
        task.parent_id = self.task_id
        self.subtasks.append(task)

    def is_executable(self, completed_tasks: set) -> bool:
        """Check if task can be executed (all dependencies met)."""
        return all(dep in completed_tasks for dep in self.dependencies)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "priority": self.priority,
        }


class GoalDecomposer:
    """Decomposes high-level goals into executable tasks."""

    def __init__(self):
        self.task_counter = 0

    def _generate_task_id(self) -> str:
        self.task_counter += 1
        return f"task_{self.task_counter}"

    def decompose(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        max_depth: int = 3,
    ) -> Task:
        """Decompose a goal into a task hierarchy."""
        root_task = Task(
            description=goal,
            task_id=self._generate_task_id(),
            metadata={"depth": 0, "context": context or {}},
        )

        # Simple heuristic decomposition based on keywords
        self._decompose_recursive(root_task, max_depth, current_depth=1)
        return root_task

    def _decompose_recursive(
        self,
        task: Task,
        max_depth: int,
        current_depth: int,
    ) -> None:
        """Recursively decompose tasks."""
        if current_depth > max_depth:
            return

        # Check for decomposable patterns
        desc_lower = task.description.lower()

        if "and" in desc_lower or "," in desc_lower:
            # Split compound goals
            parts = self._split_compound_goal(task.description)
            for i, part in enumerate(parts):
                subtask = Task(
                    description=part.strip(),
                    task_id=self._generate_task_id(),
                    priority=i + 1,
                    metadata={"depth": current_depth},
                )
                task.add_subtask(subtask)
                self._decompose_recursive(subtask, max_depth, current_depth + 1)

        elif any(word in desc_lower for word in ["analyze", "research", "study"]):
            # Research-type goals
            subtask1 = Task(
                description="Gather relevant information",
                task_id=self._generate_task_id(),
                metadata={"depth": current_depth},
            )
            subtask2 = Task(
                description="Process and synthesize findings",
                task_id=self._generate_task_id(),
                dependencies=[subtask1.task_id],
                metadata={"depth": current_depth},
            )
            subtask3 = Task(
                description="Formulate conclusions",
                task_id=self._generate_task_id(),
                dependencies=[subtask2.task_id],
                metadata={"depth": current_depth},
            )
            task.add_subtask(subtask1)
            task.add_subtask(subtask2)
            task.add_subtask(subtask3)

        elif any(word in desc_lower for word in ["create", "build", "generate"]):
            # Creation-type goals
            subtask1 = Task(
                description="Design structure and plan",
                task_id=self._generate_task_id(),
                metadata={"depth": current_depth},
            )
            subtask2 = Task(
                description="Implement core components",
                task_id=self._generate_task_id(),
                dependencies=[subtask1.task_id],
                metadata={"depth": current_depth},
            )
            subtask3 = Task(
                description="Test and refine",
                task_id=self._generate_task_id(),
                dependencies=[subtask2.task_id],
                metadata={"depth": current_depth},
            )
            task.add_subtask(subtask1)
            task.add_subtask(subtask2)
            task.add_subtask(subtask3)

    def _split_compound_goal(self, goal: str) -> List[str]:
        """Split compound goals into parts."""
        # Try splitting by 'and'
        if " and " in goal.lower():
            return [p for p in goal.split(" and ") if p.strip()]
        # Try splitting by commas
        elif "," in goal:
            return [p for p in goal.split(",") if p.strip()]
        return [goal]

    def get_execution_order(self, root_task: Task) -> List[Task]:
        """Get tasks in dependency-respecting execution order (topological sort)."""
        all_tasks = self._flatten_tasks(root_task)
        completed = set()
        ordered = []

        # Sort by priority first
        all_tasks.sort(key=lambda t: t.priority)

        while len(ordered) < len(all_tasks):
            made_progress = False
            for task in all_tasks:
                if task.task_id not in completed and task.is_executable(completed):
                    ordered.append(task)
                    completed.add(task.task_id)
                    made_progress = True

            if not made_progress:
                # Handle circular dependencies or unreachable tasks
                remaining = [t for t in all_tasks if t.task_id not in completed]
                if remaining:
                    ordered.extend(remaining)
                break

        return ordered

    def _flatten_tasks(self, task: Task) -> List[Task]:
        """Flatten task hierarchy into a list."""
        result = []
        if not task.subtasks:
            result.append(task)
        else:
            for subtask in task.subtasks:
                result.extend(self._flatten_tasks(subtask))
        return result

    def visualize_plan(self, task: Task, indent: int = 0) -> str:
        """Generate a readable task hierarchy."""
        lines = []
        prefix = "  " * indent
        status_icon = {
            TaskStatus.PENDING: "◯",
            TaskStatus.IN_PROGRESS: "◐",
            TaskStatus.COMPLETED: "✓",
            TaskStatus.FAILED: "✗",
            TaskStatus.BLOCKED: "⊗",
        }
        icon = status_icon.get(task.status, "?")

        lines.append(f"{prefix}{icon} {task.description} [{task.task_id}]")
        if task.dependencies:
            lines.append(f"{prefix}  Dependencies: {', '.join(task.dependencies)}")

        for subtask in task.subtasks:
            lines.append(self.visualize_plan(subtask, indent + 1))

        return "\n".join(lines)
