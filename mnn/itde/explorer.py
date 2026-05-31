"""mnn.itde.explorer — Knowledge Graph Explorer + Research Notebook.

Module 6: Knowledge Graph Explorer — navigate the mathematical graph,
discover cross-domain links, inspect definitions/theorems/proofs.

Module 7: Research Notebook — reproducible mathematical lab notebook
storing experiments, conjectures, proof attempts, visualizations, results.
"""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ExplorerNodeType(Enum):
    DEFINITION = "definition"
    THEOREM = "theorem"
    CONJECTURE = "conjecture"
    PROOF = "proof"
    ALGORITHM = "algorithm"
    EXAMPLE = "example"


@dataclass
class ExplorerNode:
    """A node in the knowledge graph explorer."""
    name: str
    kind: ExplorerNodeType
    domain: str
    content: str
    tags: List[str] = field(default_factory=list)


@dataclass
class ExplorerEdge:
    """An edge in the knowledge graph explorer."""
    source: str
    target: str
    relation: str  # implies, generalizes, specializes, depends_on, equivalent_to


class KnowledgeGraphExplorer:
    """Module 6: Interactive knowledge graph exploration."""

    def __init__(self):
        self.nodes: Dict[str, ExplorerNode] = {}
        self.edges: List[ExplorerEdge] = []
        self._fwd: Dict[str, List[Tuple[str, str]]] = {}
        self._rev: Dict[str, List[Tuple[str, str]]] = {}

    def add_node(self, name: str, kind: str, domain: str, content: str,
                  tags: List[str] = None):
        ek = ExplorerNodeType(kind) if kind in [e.value for e in ExplorerNodeType] else ExplorerNodeType.DEFINITION
        node = ExplorerNode(name, ek, domain, content, tags or [])
        self.nodes[name] = node
        if name not in self._fwd:
            self._fwd[name] = []
            self._rev[name] = []

    def add_edge(self, source: str, target: str, relation: str):
        self.edges.append(ExplorerEdge(source, target, relation))
        for n in [source, target]:
            if n not in self._fwd:
                self._fwd[n] = []
            if n not in self._rev:
                self._rev[n] = []
        self._fwd[source].append((target, relation))
        self._rev[target].append((source, relation))

    def explore(self, name: str) -> Dict:
        """Explore a node: see content, connections, related domains."""
        node = self.nodes.get(name)
        if not node:
            return {"error": f"'{name}' not found"}
        forward = self._fwd.get(name, [])
        backward = self._rev.get(name, [])
        related_domains = set()
        for t, _ in forward:
            n = self.nodes.get(t)
            if n and n.domain != node.domain:
                related_domains.add(n.domain)
        for s, _ in backward:
            n = self.nodes.get(s)
            if n and n.domain != node.domain:
                related_domains.add(n.domain)
        return {
            "name": name, "kind": node.kind.value, "domain": node.domain,
            "content": node.content, "tags": node.tags,
            "forward_links": [(t, r) for t, r in forward],
            "backward_links": [(s, r) for s, r in backward],
            "cross_domain": list(related_domains),
        }

    def search(self, query: str) -> List[str]:
        q = query.lower()
        return [n for n, node in self.nodes.items()
                if q in n.lower() or q in node.content.lower()
                or q in node.domain.lower() or any(q in t for t in node.tags)]

    def by_domain(self, domain: str) -> List[str]:
        return [n for n, node in self.nodes.items() if node.domain == domain]

    def cross_domain_connections(self) -> List[Dict]:
        links = []
        for e in self.edges:
            s, t = self.nodes.get(e.source), self.nodes.get(e.target)
            if s and t and s.domain != t.domain:
                links.append({"source": e.source, "target": e.target,
                              "relation": e.relation,
                              "domains": [s.domain, t.domain]})
        return links

    def neighborhood(self, name: str, depth: int = 2) -> Dict[str, int]:
        """BFS neighborhood up to given depth."""
        visited = {}
        queue = [(name, 0)]
        while queue:
            current, d = queue.pop(0)
            if current in visited or d > depth:
                continue
            visited[current] = d
            for t, _ in self._fwd.get(current, []):
                queue.append((t, d + 1))
            for s, _ in self._rev.get(current, []):
                queue.append((s, d + 1))
        return visited

    def summary(self) -> str:
        domains = set(n.domain for n in self.nodes.values())
        return (f"KGExplorer: {len(self.nodes)} nodes, {len(self.edges)} edges, "
                f"{len(domains)} domains")


# ---- Module 7: Research Notebook ----

class CellType(Enum):
    MARKDOWN = "markdown"
    CODE = "code"
    RESULT = "result"
    CONJECTURE = "conjecture"
    PROOF = "proof"
    VISUALIZATION = "visualization"
    EXPERIMENT = "experiment"


@dataclass
class NotebookCell:
    """A cell in the research notebook."""
    cell_type: CellType
    content: str
    output: str = ""
    metadata: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class ResearchNotebook:
    """Module 7: Reproducible mathematical lab notebook."""

    def __init__(self, title: str = "Research Notebook"):
        self.title = title
        self.cells: List[NotebookCell] = []
        self.tags: List[str] = []

    def add_markdown(self, content: str) -> int:
        self.cells.append(NotebookCell(CellType.MARKDOWN, content))
        return len(self.cells) - 1

    def add_code(self, code: str, output: str = "") -> int:
        self.cells.append(NotebookCell(CellType.CODE, code, output))
        return len(self.cells) - 1

    def add_experiment(self, description: str, result: str = "") -> int:
        self.cells.append(NotebookCell(CellType.EXPERIMENT, description, result))
        return len(self.cells) - 1

    def add_conjecture(self, statement: str, status: str = "open") -> int:
        cell = NotebookCell(CellType.CONJECTURE, statement)
        cell.metadata["status"] = status
        self.cells.append(cell)
        return len(self.cells) - 1

    def add_proof(self, theorem: str, proof: str) -> int:
        self.cells.append(NotebookCell(CellType.PROOF, f"{theorem}\n---\n{proof}"))
        return len(self.cells) - 1

    def add_result(self, result: str) -> int:
        self.cells.append(NotebookCell(CellType.RESULT, result))
        return len(self.cells) - 1

    def render(self) -> str:
        lines = [f"{'=' * 60}", f"  {self.title}", f"{'=' * 60}"]
        for i, cell in enumerate(self.cells):
            lines.append(f"\n--- Cell [{i}] ({cell.cell_type.value}) ---")
            lines.append(cell.content)
            if cell.output:
                lines.append(f"  Output: {cell.output}")
        return "\n".join(lines)

    def export_dict(self) -> Dict:
        return {
            "title": self.title,
            "cells": [{
                "type": c.cell_type.value, "content": c.content,
                "output": c.output, "metadata": c.metadata
            } for c in self.cells],
        }

    def summary(self) -> str:
        counts = {}
        for c in self.cells:
            counts[c.cell_type.value] = counts.get(c.cell_type.value, 0) + 1
        return f"Notebook({self.title}): {len(self.cells)} cells, types={counts}"
