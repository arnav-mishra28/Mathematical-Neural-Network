"""mnn.researcher.literature — Literature Engine + Research Knowledge Graph.

Module 1: Literature Engine — ingest papers, textbooks, theorem libraries.
Build structured concept graphs from sources.

Module 2: Research Knowledge Graph — connect theorems, proofs, equations,
operators with rich edge semantics (implies, generalizes, contradicts, depends).
"""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


class SourceType(Enum):
    PAPER = "paper"
    TEXTBOOK = "textbook"
    THEOREM_LIBRARY = "theorem_library"
    DATASET = "dataset"
    LECTURE = "lecture"
    PREPRINT = "preprint"


class RelationType(Enum):
    IMPLIES = "implies"
    GENERALIZES = "generalizes"
    SPECIALIZES = "specializes"
    CONTRADICTS = "contradicts"
    DEPENDS_ON = "depends_on"
    EXTENDS = "extends"
    EQUIVALENT = "equivalent"
    RELATED = "related"
    USES = "uses"
    MOTIVATES = "motivates"


@dataclass
class LiteratureSource:
    """A single literature source."""
    title: str
    source_type: SourceType
    authors: List[str] = field(default_factory=list)
    year: int = 0
    abstract: str = ""
    key_concepts: List[str] = field(default_factory=list)
    theorems: List[str] = field(default_factory=list)
    open_problems: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class LiteratureEngine:
    """Module 1: Manage and analyze mathematical literature."""

    def __init__(self):
        self.sources: Dict[str, LiteratureSource] = {}
        self.concept_index: Dict[str, List[str]] = {}  # concept → source titles

    def add_source(self, source: LiteratureSource):
        self.sources[source.title] = source
        for concept in source.key_concepts:
            if concept not in self.concept_index:
                self.concept_index[concept] = []
            self.concept_index[concept].append(source.title)

    def search(self, query: str) -> List[LiteratureSource]:
        q = query.lower()
        return [s for s in self.sources.values()
                if q in s.title.lower() or q in s.abstract.lower()
                or any(q in c.lower() for c in s.key_concepts)]

    def by_concept(self, concept: str) -> List[LiteratureSource]:
        titles = self.concept_index.get(concept, [])
        return [self.sources[t] for t in titles if t in self.sources]

    def open_problems(self) -> List[Dict]:
        problems = []
        for s in self.sources.values():
            for p in s.open_problems:
                problems.append({"problem": p, "source": s.title, "year": s.year})
        return problems

    def concept_map(self) -> Dict[str, int]:
        """Frequency of each concept across all sources."""
        return {c: len(titles) for c, titles in self.concept_index.items()}

    def citation_graph(self) -> Dict[str, List[str]]:
        """Build citation graph from references."""
        graph = {}
        for title, source in self.sources.items():
            graph[title] = [r for r in source.references if r in self.sources]
        return graph

    def summary(self) -> str:
        return (f"LiteratureEngine: {len(self.sources)} sources, "
                f"{len(self.concept_index)} concepts, "
                f"{sum(len(s.open_problems) for s in self.sources.values())} open problems")


# ---- Module 2: Research Knowledge Graph ----

@dataclass
class ResearchNode:
    """A node in the research knowledge graph."""
    name: str
    kind: str  # theorem, proof, equation, operator, dataset, conjecture
    domain: str
    content: str
    confidence: float = 1.0
    source: str = ""
    tags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ResearchEdge:
    """A directed edge between research nodes."""
    source: str
    target: str
    relation: RelationType
    weight: float = 1.0
    evidence: str = ""


class ResearchKnowledgeGraph:
    """Module 2: Rich research knowledge graph with typed edges."""

    def __init__(self):
        self.nodes: Dict[str, ResearchNode] = {}
        self.edges: List[ResearchEdge] = []
        self._fwd: Dict[str, List[Tuple[str, RelationType]]] = {}
        self._rev: Dict[str, List[Tuple[str, RelationType]]] = {}

    def add_node(self, node: ResearchNode):
        self.nodes[node.name] = node
        if node.name not in self._fwd:
            self._fwd[node.name] = []
            self._rev[node.name] = []

    def add_edge(self, edge: ResearchEdge):
        self.edges.append(edge)
        for n in [edge.source, edge.target]:
            if n not in self._fwd:
                self._fwd[n] = []
            if n not in self._rev:
                self._rev[n] = []
        self._fwd[edge.source].append((edge.target, edge.relation))
        self._rev[edge.target].append((edge.source, edge.relation))

    def neighbors(self, name: str, relation: RelationType = None) -> List[str]:
        edges = self._fwd.get(name, [])
        if relation:
            return [t for t, r in edges if r == relation]
        return [t for t, _ in edges]

    def predecessors(self, name: str, relation: RelationType = None) -> List[str]:
        edges = self._rev.get(name, [])
        if relation:
            return [s for s, r in edges if r == relation]
        return [s for s, _ in edges]

    def find_contradictions(self) -> List[Tuple[str, str]]:
        return [(e.source, e.target) for e in self.edges
                if e.relation == RelationType.CONTRADICTS]

    def find_path(self, start: str, end: str, max_depth: int = 10) -> Optional[List[str]]:
        """BFS shortest path."""
        if start == end:
            return [start]
        visited = {start}
        queue = [(start, [start])]
        for _ in range(max_depth * len(self.nodes)):
            if not queue:
                break
            current, path = queue.pop(0)
            for nb in [t for t, _ in self._fwd.get(current, [])]:
                if nb == end:
                    return path + [nb]
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return None

    def cross_domain_links(self) -> List[Dict]:
        """Find edges connecting different domains."""
        links = []
        for e in self.edges:
            s_node = self.nodes.get(e.source)
            t_node = self.nodes.get(e.target)
            if s_node and t_node and s_node.domain != t_node.domain:
                links.append({
                    "source": e.source, "target": e.target,
                    "source_domain": s_node.domain, "target_domain": t_node.domain,
                    "relation": e.relation.value,
                })
        return links

    def summary(self) -> str:
        rel_counts = {}
        for e in self.edges:
            rel_counts[e.relation.value] = rel_counts.get(e.relation.value, 0) + 1
        return (f"ResearchKG: {len(self.nodes)} nodes, {len(self.edges)} edges, "
                f"relations={rel_counts}")
