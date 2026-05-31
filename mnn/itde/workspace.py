"""mnn.itde.workspace — Mathematical Workspace + Live Theorem Canvas.

Module 1: Mathematical Workspace — create and inspect math objects
(groups, manifolds, tensors, PDEs, theorem graphs) with auto-properties.

Module 2: Live Theorem Canvas — visual theorem graph structure with
clickable nodes, dependency inspection, and proof idea tracking.
"""
from __future__ import annotations
import numpy as np
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ObjectType(Enum):
    GROUP = "group"
    RING = "ring"
    FIELD = "field"
    MANIFOLD = "manifold"
    TENSOR = "tensor"
    PDE = "pde"
    GRAPH = "graph"
    OPERATOR = "operator"
    FUNCTION = "function"
    SPACE = "space"
    CUSTOM = "custom"


@dataclass
class MathObject:
    """A mathematical object in the workspace."""
    name: str
    obj_type: ObjectType
    properties: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    relations: List[str] = field(default_factory=list)
    computed: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def info(self) -> str:
        lines = [f"╔═ {self.name} ({self.obj_type.value})",
                 f"║  {self.description}" if self.description else ""]
        if self.properties:
            lines.append("║  Properties:")
            for k, v in self.properties.items():
                lines.append(f"║    {k}: {v}")
        if self.computed:
            lines.append("║  Computed:")
            for k, v in self.computed.items():
                lines.append(f"║    {k}: {v}")
        if self.relations:
            lines.append(f"║  Relations: {', '.join(self.relations)}")
        lines.append("╚═")
        return "\n".join(l for l in lines if l)


class MathWorkspace:
    """Module 1: Interactive mathematical workspace."""

    def __init__(self, name: str = "Workspace"):
        self.name = name
        self.objects: Dict[str, MathObject] = {}
        self.history: List[Dict] = []
        self._auto_properties: Dict[ObjectType, Callable] = {}
        self._register_auto()

    def _register_auto(self):
        self._auto_properties[ObjectType.GROUP] = self._auto_group
        self._auto_properties[ObjectType.MANIFOLD] = self._auto_manifold
        self._auto_properties[ObjectType.PDE] = self._auto_pde

    def create(self, name: str, obj_type: str, description: str = "",
               properties: Dict = None, **kwargs) -> MathObject:
        """Create a mathematical object."""
        ot = ObjectType(obj_type) if obj_type in [e.value for e in ObjectType] else ObjectType.CUSTOM
        obj = MathObject(name, ot, properties or {}, description)
        obj.properties.update(kwargs)
        # Auto-compute
        auto_fn = self._auto_properties.get(ot)
        if auto_fn:
            auto_fn(obj)
        self.objects[name] = obj
        self.history.append({"action": "create", "name": name, "type": ot.value,
                             "timestamp": time.time()})
        return obj

    def get(self, name: str) -> Optional[MathObject]:
        return self.objects.get(name)

    def inspect(self, name: str) -> str:
        obj = self.objects.get(name)
        return obj.info() if obj else f"Object '{name}' not found."

    def connect(self, a: str, b: str, relation: str = "related"):
        """Connect two objects."""
        for n in [a, b]:
            if n in self.objects:
                self.objects[n].relations.append(f"{relation}→{b if n == a else a}")

    def search(self, query: str) -> List[MathObject]:
        q = query.lower()
        return [o for o in self.objects.values()
                if q in o.name.lower() or q in o.description.lower()
                or q in o.obj_type.value]

    def list_objects(self) -> List[str]:
        return [f"{n} ({o.obj_type.value})" for n, o in self.objects.items()]

    def _auto_group(self, obj: MathObject):
        order = obj.properties.get("order")
        if order:
            obj.computed["is_finite"] = True
            obj.computed["order"] = order
            obj.computed["is_abelian"] = obj.properties.get("abelian", "unknown")
            if isinstance(order, int):
                factors = []
                n = order
                for p in [2, 3, 5, 7, 11, 13]:
                    while n % p == 0:
                        factors.append(p)
                        n //= p
                if n > 1:
                    factors.append(n)
                obj.computed["prime_factorization"] = factors

    def _auto_manifold(self, obj: MathObject):
        dim = obj.properties.get("dimension")
        if dim:
            obj.computed["dimension"] = dim
        curvature = obj.properties.get("curvature")
        if curvature is not None:
            obj.computed["curvature_type"] = (
                "flat" if curvature == 0 else "positive" if curvature > 0 else "negative")
        radius = obj.properties.get("radius")
        if radius and dim:
            if dim == 2:
                obj.computed["area"] = 4 * np.pi * radius**2
                obj.computed["euler_characteristic"] = 2

    def _auto_pde(self, obj: MathObject):
        pde_type = obj.properties.get("type")
        if pde_type:
            obj.computed["classification"] = pde_type
        order = obj.properties.get("order")
        if order:
            obj.computed["order"] = order

    def summary(self) -> str:
        type_counts = {}
        for o in self.objects.values():
            type_counts[o.obj_type.value] = type_counts.get(o.obj_type.value, 0) + 1
        return f"Workspace({self.name}): {len(self.objects)} objects, types={type_counts}"


# ---- Module 2: Live Theorem Canvas ----

class NodeStatus(Enum):
    UNVERIFIED = "unverified"
    CONJECTURED = "conjectured"
    PROVED = "proved"
    PARTIALLY_PROVED = "partially_proved"
    DISPROVED = "disproved"


@dataclass
class CanvasNode:
    """A node in the theorem canvas."""
    name: str
    kind: str  # definition, lemma, theorem, conjecture
    statement: str
    status: NodeStatus = NodeStatus.UNVERIFIED
    assumptions: List[str] = field(default_factory=list)
    proof_ideas: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class CanvasEdge:
    """Edge between canvas nodes."""
    source: str
    target: str
    label: str = "depends"


class TheoremCanvas:
    """Module 2: Interactive theorem graph visualization."""

    def __init__(self):
        self.nodes: Dict[str, CanvasNode] = {}
        self.edges: List[CanvasEdge] = []

    def add_definition(self, name: str, statement: str) -> CanvasNode:
        node = CanvasNode(name, "definition", statement, NodeStatus.PROVED)
        self.nodes[name] = node
        return node

    def add_lemma(self, name: str, statement: str,
                   depends: List[str] = None) -> CanvasNode:
        node = CanvasNode(name, "lemma", statement)
        self.nodes[name] = node
        for dep in (depends or []):
            self.edges.append(CanvasEdge(dep, name, "uses"))
        return node

    def add_theorem(self, name: str, statement: str,
                     depends: List[str] = None) -> CanvasNode:
        node = CanvasNode(name, "theorem", statement)
        self.nodes[name] = node
        for dep in (depends or []):
            self.edges.append(CanvasEdge(dep, name, "uses"))
        return node

    def add_conjecture(self, name: str, statement: str,
                        depends: List[str] = None) -> CanvasNode:
        node = CanvasNode(name, "conjecture", statement, NodeStatus.CONJECTURED)
        self.nodes[name] = node
        for dep in (depends or []):
            self.edges.append(CanvasEdge(dep, name, "motivates"))
        return node

    def set_status(self, name: str, status: str):
        if name in self.nodes:
            self.nodes[name].status = NodeStatus(status)

    def add_proof_idea(self, name: str, idea: str):
        if name in self.nodes:
            self.nodes[name].proof_ideas.append(idea)

    def inspect_node(self, name: str) -> Dict:
        node = self.nodes.get(name)
        if not node:
            return {"error": f"Node '{name}' not found"}
        deps = [e.source for e in self.edges if e.target == name]
        dependents = [e.target for e in self.edges if e.source == name]
        return {
            "name": node.name, "kind": node.kind,
            "statement": node.statement, "status": node.status.value,
            "assumptions": node.assumptions,
            "proof_ideas": node.proof_ideas,
            "dependencies": deps, "dependents": dependents,
        }

    def render_tree(self) -> str:
        """Render theorem graph as text tree."""
        roots = set(self.nodes.keys()) - {e.target for e in self.edges}
        lines = []
        visited = set()
        for root in sorted(roots):
            self._render_subtree(root, 0, lines, visited)
        return "\n".join(lines) if lines else "(empty canvas)"

    def _render_subtree(self, name: str, depth: int, lines: List, visited: set):
        if name in visited or name not in self.nodes:
            return
        visited.add(name)
        node = self.nodes[name]
        status_sym = {"unverified": "○", "conjectured": "?", "proved": "●",
                      "partially_proved": "◑", "disproved": "✗"}
        sym = status_sym.get(node.status.value, "·")
        lines.append(f"{'  ' * depth}{sym} [{node.kind}] {name}: {node.statement[:60]}")
        children = [e.target for e in self.edges if e.source == name]
        for child in children:
            self._render_subtree(child, depth + 1, lines, visited)

    def summary(self) -> str:
        kind_counts = {}
        for n in self.nodes.values():
            kind_counts[n.kind] = kind_counts.get(n.kind, 0) + 1
        return f"TheoremCanvas: {len(self.nodes)} nodes, {len(self.edges)} edges, kinds={kind_counts}"
