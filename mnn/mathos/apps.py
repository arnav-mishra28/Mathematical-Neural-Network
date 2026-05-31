"""mnn.mathos.apps — Application Ecosystem (Layer 9) + Mathematical Cloud (Layer 10).

Layer 9: Application Ecosystem — research apps, education apps, engineering apps,
physics apps, AI apps built on top of MathOS.

Layer 10: Mathematical Cloud — distributed network of researchers, agents,
theorem databases, and simulations.
"""
from __future__ import annotations
import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


# ============= Layer 9: Application Ecosystem =============

class AppCategory(Enum):
    RESEARCH = "research"
    EDUCATION = "education"
    ENGINEERING = "engineering"
    PHYSICS = "physics"
    AI = "ai"
    VISUALIZATION = "visualization"
    CUSTOM = "custom"


@dataclass
class MathApp:
    """An application running on MathOS."""
    name: str
    category: AppCategory
    description: str
    version: str = "1.0"
    entrypoint: Optional[Callable] = None
    config: Dict = field(default_factory=dict)
    status: str = "installed"  # installed, running, stopped, error

    def run(self, **kwargs) -> Any:
        self.status = "running"
        if self.entrypoint:
            try:
                result = self.entrypoint(**kwargs)
                self.status = "installed"
                return result
            except Exception as e:
                self.status = "error"
                return {"error": str(e)}
        self.status = "installed"
        return {"message": f"App '{self.name}' has no entrypoint."}


class AppEcosystem:
    """Layer 9: Mathematical application platform."""

    def __init__(self):
        self.apps: Dict[str, MathApp] = {}
        self._register_builtin()

    def _register_builtin(self):
        self.install(MathApp(
            "TheoremExplorer", AppCategory.RESEARCH,
            "Explore theorem databases and discover connections"))
        self.install(MathApp(
            "MathTutor", AppCategory.EDUCATION,
            "Interactive mathematics tutor with multi-level explanations"))
        self.install(MathApp(
            "PDESimulator", AppCategory.ENGINEERING,
            "Simulate and analyze partial differential equations"))
        self.install(MathApp(
            "OperatorLearner", AppCategory.PHYSICS,
            "Learn physical operators from data"))
        self.install(MathApp(
            "SymbolicReasoner", AppCategory.AI,
            "AI-powered symbolic reasoning and proof search"))
        self.install(MathApp(
            "ManifoldViewer", AppCategory.VISUALIZATION,
            "Interactive 3D manifold visualization"))

    def install(self, app: MathApp):
        self.apps[app.name] = app

    def uninstall(self, name: str) -> bool:
        return self.apps.pop(name, None) is not None

    def run_app(self, name: str, **kwargs) -> Any:
        app = self.apps.get(name)
        if app:
            return app.run(**kwargs)
        return {"error": f"App '{name}' not found"}

    def by_category(self, category: str) -> List[MathApp]:
        cat = AppCategory(category) if category in [e.value for e in AppCategory] else AppCategory.CUSTOM
        return [a for a in self.apps.values() if a.category == cat]

    def list_apps(self) -> List[Dict]:
        return [{"name": a.name, "category": a.category.value,
                 "description": a.description, "status": a.status}
                for a in self.apps.values()]

    def summary(self) -> str:
        cats = {}
        for a in self.apps.values():
            cats[a.category.value] = cats.get(a.category.value, 0) + 1
        return f"AppEcosystem: {len(self.apps)} apps, categories={cats}"


# ============= Layer 10: Mathematical Cloud =============

@dataclass
class CloudNode:
    """A node in the mathematical cloud network."""
    node_id: str
    name: str
    capabilities: List[str] = field(default_factory=list)
    status: str = "online"
    resources: Dict = field(default_factory=dict)
    last_seen: float = field(default_factory=time.time)


@dataclass
class CloudTask:
    """A task distributed across the cloud."""
    task_id: str
    description: str
    assigned_to: str = ""
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict] = None
    timestamp: float = field(default_factory=time.time)


class MathCloud:
    """Layer 10: Distributed network for mathematical intelligence."""

    def __init__(self, name: str = "MathCloud"):
        self.name = name
        self.nodes: Dict[str, CloudNode] = {}
        self.tasks: List[CloudTask] = []
        self.shared_knowledge: Dict[str, Any] = {}
        # Register self as a local node
        self.register_node("local", "LocalNode",
                            ["simulation", "proof", "discovery"])

    def register_node(self, node_id: str, name: str,
                       capabilities: List[str] = None):
        self.nodes[node_id] = CloudNode(node_id, name, capabilities or [])

    def submit_task(self, description: str, required_capability: str = "") -> CloudTask:
        """Submit a task to the cloud."""
        task_id = f"task_{len(self.tasks)}_{int(time.time() * 1000) % 10000}"
        task = CloudTask(task_id, description)

        # Find a node with the right capability
        for nid, node in self.nodes.items():
            if node.status == "online" and (
                    not required_capability or required_capability in node.capabilities):
                task.assigned_to = nid
                task.status = "running"
                break

        self.tasks.append(task)
        return task

    def complete_task(self, task_id: str, result: Dict = None):
        for t in self.tasks:
            if t.task_id == task_id:
                t.status = "completed"
                t.result = result or {}
                break

    def share_knowledge(self, key: str, value: Any):
        self.shared_knowledge[key] = value

    def get_shared(self, key: str) -> Optional[Any]:
        return self.shared_knowledge.get(key)

    def online_nodes(self) -> List[CloudNode]:
        return [n for n in self.nodes.values() if n.status == "online"]

    def summary(self) -> str:
        n_online = sum(1 for n in self.nodes.values() if n.status == "online")
        n_completed = sum(1 for t in self.tasks if t.status == "completed")
        return (f"MathCloud({self.name}): {len(self.nodes)} nodes ({n_online} online), "
                f"{len(self.tasks)} tasks ({n_completed} completed), "
                f"{len(self.shared_knowledge)} shared items")
