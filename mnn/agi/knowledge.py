"""mnn.agi.knowledge — Knowledge Layer.

Module 1: Stores mathematical knowledge as interconnected graphs.
Definition graphs, theorem graphs, proof graphs, operator graphs.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json, time


class KnowledgeType(Enum):
    DEFINITION = "definition"
    THEOREM = "theorem"
    LEMMA = "lemma"
    COROLLARY = "corollary"
    AXIOM = "axiom"
    CONJECTURE = "conjecture"
    OPERATOR = "operator"
    STRUCTURE = "structure"
    PROOF = "proof"
    EXAMPLE = "example"


class Domain(Enum):
    ALGEBRA = "algebra"
    TOPOLOGY = "topology"
    ANALYSIS = "analysis"
    NUMBER_THEORY = "number_theory"
    GEOMETRY = "geometry"
    CATEGORY_THEORY = "category_theory"
    PDE = "pde"
    PROBABILITY = "probability"
    LOGIC = "logic"
    COMBINATORICS = "combinatorics"


@dataclass
class KnowledgeNode:
    """A node in the mathematical knowledge graph."""
    name: str
    kind: KnowledgeType
    domain: Domain
    statement: str
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __repr__(self):
        return f"KnowledgeNode({self.kind.value}: {self.name})"


@dataclass
class KnowledgeEdge:
    """Directed edge between knowledge nodes."""
    source: str
    target: str
    relation: str  # "implies", "uses", "generalizes", "specializes", "equivalent"
    weight: float = 1.0
    metadata: Dict = field(default_factory=dict)


class MathKnowledgeGraph:
    """Mathematical knowledge graph — stores definitions, theorems, proofs as a graph."""

    def __init__(self, name: str = "MathKG"):
        self.name = name
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self._adj: Dict[str, List[str]] = {}
        self._rev_adj: Dict[str, List[str]] = {}

    def add_node(self, node: KnowledgeNode):
        self.nodes[node.name] = node
        if node.name not in self._adj:
            self._adj[node.name] = []
            self._rev_adj[node.name] = []
        for dep in node.dependencies:
            self.add_edge(KnowledgeEdge(dep, node.name, "uses"))

    def add_edge(self, edge: KnowledgeEdge):
        self.edges.append(edge)
        if edge.source not in self._adj:
            self._adj[edge.source] = []
        if edge.target not in self._rev_adj:
            self._rev_adj[edge.target] = []
        self._adj[edge.source].append(edge.target)
        self._rev_adj[edge.target].append(edge.source)

    def get_node(self, name: str) -> Optional[KnowledgeNode]:
        return self.nodes.get(name)

    def neighbors(self, name: str) -> List[str]:
        return self._adj.get(name, [])

    def predecessors(self, name: str) -> List[str]:
        return self._rev_adj.get(name, [])

    def by_domain(self, domain: Domain) -> List[KnowledgeNode]:
        return [n for n in self.nodes.values() if n.domain == domain]

    def by_type(self, kind: KnowledgeType) -> List[KnowledgeNode]:
        return [n for n in self.nodes.values() if n.kind == kind]

    def search(self, query: str) -> List[KnowledgeNode]:
        q = query.lower()
        return [n for n in self.nodes.values()
                if q in n.name.lower() or q in n.statement.lower()
                or any(q in t for t in n.tags)]

    def dependency_chain(self, name: str) -> List[str]:
        """BFS to find all transitive dependencies."""
        visited = set()
        queue = [name]
        chain = []
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            chain.append(current)
            queue.extend(self._rev_adj.get(current, []))
        return chain

    def dependents(self, name: str) -> List[str]:
        """Everything that depends on this node."""
        visited = set()
        queue = [name]
        deps = []
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            deps.append(current)
            queue.extend(self._adj.get(current, []))
        return deps

    def topological_sort(self) -> List[str]:
        """Topological order of the knowledge graph."""
        in_degree = {n: 0 for n in self.nodes}
        for e in self.edges:
            if e.target in in_degree:
                in_degree[e.target] += 1
        queue = [n for n, d in in_degree.items() if d == 0]
        result = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for nb in self._adj.get(node, []):
                if nb in in_degree:
                    in_degree[nb] -= 1
                    if in_degree[nb] == 0:
                        queue.append(nb)
        return result

    def adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        names = list(self.nodes.keys())
        idx = {n: i for i, n in enumerate(names)}
        n = len(names)
        A = np.zeros((n, n))
        for e in self.edges:
            if e.source in idx and e.target in idx:
                A[idx[e.source], idx[e.target]] = e.weight
        return A, names

    def summary(self) -> str:
        type_counts = {}
        for n in self.nodes.values():
            type_counts[n.kind.value] = type_counts.get(n.kind.value, 0) + 1
        return (f"MathKnowledgeGraph({self.name}): "
                f"{len(self.nodes)} nodes, {len(self.edges)} edges, "
                f"types={type_counts}")

    def add_standard_knowledge(self):
        """Seed with standard mathematical knowledge."""
        entries = [
            ("group_def", KnowledgeType.DEFINITION, Domain.ALGEBRA,
             "A group (G,*) is a set G with operation * satisfying closure, associativity, identity, and inverse."),
            ("ring_def", KnowledgeType.DEFINITION, Domain.ALGEBRA,
             "A ring (R,+,*) is an abelian group under + with associative * and distributive laws.", ["group_def"]),
            ("field_def", KnowledgeType.DEFINITION, Domain.ALGEBRA,
             "A field is a commutative ring where every nonzero element has a multiplicative inverse.", ["ring_def"]),
            ("vector_space_def", KnowledgeType.DEFINITION, Domain.ALGEBRA,
             "A vector space V over field F is an abelian group with scalar multiplication.", ["field_def", "group_def"]),
            ("topology_def", KnowledgeType.DEFINITION, Domain.TOPOLOGY,
             "A topology on set X is a collection of open sets closed under union and finite intersection."),
            ("continuous_def", KnowledgeType.DEFINITION, Domain.TOPOLOGY,
             "f:X→Y is continuous if preimage of every open set is open.", ["topology_def"]),
            ("homeomorphism_def", KnowledgeType.DEFINITION, Domain.TOPOLOGY,
             "A homeomorphism is a continuous bijection with continuous inverse.", ["continuous_def"]),
            ("metric_space_def", KnowledgeType.DEFINITION, Domain.ANALYSIS,
             "A metric space (X,d) has distance d satisfying positivity, symmetry, triangle inequality."),
            ("cauchy_thm", KnowledgeType.THEOREM, Domain.ANALYSIS,
             "Every Cauchy sequence in a complete metric space converges.", ["metric_space_def"]),
            ("lagrange_thm", KnowledgeType.THEOREM, Domain.ALGEBRA,
             "Order of subgroup divides order of finite group.", ["group_def"]),
            ("ftoa", KnowledgeType.THEOREM, Domain.ALGEBRA,
             "Every non-constant polynomial over C has a root.", ["field_def"]),
            ("heat_eq", KnowledgeType.DEFINITION, Domain.PDE,
             "∂u/∂t = α∇²u — models heat diffusion."),
            ("wave_eq", KnowledgeType.DEFINITION, Domain.PDE,
             "∂²u/∂t² = c²∇²u — models wave propagation."),
        ]
        for entry in entries:
            name, kind, domain, stmt = entry[0], entry[1], entry[2], entry[3]
            deps = entry[4] if len(entry) > 4 else []
            self.add_node(KnowledgeNode(name, kind, domain, stmt, deps,
                                         tags=[domain.value]))
