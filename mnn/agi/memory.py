"""mnn.agi.memory — Mathematical Memory.

Module 2: Specialized memory beyond standard LLM memory.
Concept Memory (definitions, objects, structures),
Proof Memory (strategies, transformations, reusable arguments),
Research Memory (conjectures, failed approaches, discovered patterns).
"""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class MemoryType(Enum):
    CONCEPT = "concept"
    PROOF = "proof"
    RESEARCH = "research"
    OBSERVATION = "observation"
    STRATEGY = "strategy"


@dataclass
class MemoryEntry:
    """A single memory entry."""
    content: str
    memory_type: MemoryType
    tags: List[str] = field(default_factory=list)
    importance: float = 1.0
    access_count: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    def access(self):
        self.access_count += 1


class ConceptMemory:
    """Stores definitions, objects, structures."""

    def __init__(self):
        self.entries: Dict[str, MemoryEntry] = {}

    def store(self, name: str, definition: str, tags: List[str] = None,
              importance: float = 1.0):
        self.entries[name] = MemoryEntry(
            definition, MemoryType.CONCEPT, tags or [], importance)

    def recall(self, name: str) -> Optional[str]:
        entry = self.entries.get(name)
        if entry:
            entry.access()
            return entry.content
        return None

    def search(self, query: str) -> List[str]:
        q = query.lower()
        results = []
        for name, entry in self.entries.items():
            if q in name.lower() or q in entry.content.lower() or any(q in t for t in entry.tags):
                results.append(name)
        return results

    def most_accessed(self, top_k: int = 10) -> List[str]:
        sorted_entries = sorted(self.entries.items(), key=lambda x: -x[1].access_count)
        return [name for name, _ in sorted_entries[:top_k]]


class ProofMemory:
    """Stores proof strategies, transformations, reusable arguments."""

    def __init__(self):
        self.strategies: List[MemoryEntry] = []
        self.transformations: List[MemoryEntry] = []
        self.arguments: Dict[str, MemoryEntry] = {}

    def store_strategy(self, name: str, description: str,
                        applicable_domains: List[str] = None):
        entry = MemoryEntry(description, MemoryType.STRATEGY,
                             applicable_domains or [], 1.0)
        entry.metadata["name"] = name
        self.strategies.append(entry)

    def store_argument(self, name: str, argument: str,
                        tags: List[str] = None):
        self.arguments[name] = MemoryEntry(
            argument, MemoryType.PROOF, tags or [])

    def recall_strategies(self, domain: str = None) -> List[Dict]:
        results = []
        for s in self.strategies:
            if domain is None or domain in s.tags:
                s.access()
                results.append({"name": s.metadata.get("name", ""),
                                "description": s.content,
                                "domains": s.tags})
        return results

    def recall_argument(self, name: str) -> Optional[str]:
        entry = self.arguments.get(name)
        if entry:
            entry.access()
            return entry.content
        return None


class ResearchMemory:
    """Stores conjectures, failed approaches, discovered patterns."""

    def __init__(self):
        self.conjectures: List[MemoryEntry] = []
        self.failed_approaches: List[MemoryEntry] = []
        self.patterns: List[MemoryEntry] = []
        self.insights: List[MemoryEntry] = []

    def add_conjecture(self, statement: str, confidence: float = 0.5,
                        tags: List[str] = None):
        entry = MemoryEntry(statement, MemoryType.RESEARCH, tags or [], confidence)
        entry.metadata["status"] = "open"
        self.conjectures.append(entry)

    def add_failed_approach(self, description: str, reason: str,
                              tags: List[str] = None):
        entry = MemoryEntry(description, MemoryType.RESEARCH, tags or [])
        entry.metadata["reason"] = reason
        self.failed_approaches.append(entry)

    def add_pattern(self, pattern: str, evidence: str = "",
                     tags: List[str] = None):
        entry = MemoryEntry(pattern, MemoryType.OBSERVATION, tags or [])
        entry.metadata["evidence"] = evidence
        self.patterns.append(entry)

    def add_insight(self, insight: str, tags: List[str] = None):
        self.insights.append(MemoryEntry(insight, MemoryType.RESEARCH, tags or []))

    def open_conjectures(self) -> List[str]:
        return [c.content for c in self.conjectures
                if c.metadata.get("status") == "open"]

    def resolve_conjecture(self, index: int, status: str = "proved"):
        if 0 <= index < len(self.conjectures):
            self.conjectures[index].metadata["status"] = status

    def search_patterns(self, query: str) -> List[str]:
        q = query.lower()
        return [p.content for p in self.patterns
                if q in p.content.lower() or any(q in t for t in p.tags)]


class MathematicalMemory:
    """Unified mathematical memory system."""

    def __init__(self):
        self.concepts = ConceptMemory()
        self.proofs = ProofMemory()
        self.research = ResearchMemory()

    def summary(self) -> Dict:
        return {
            "concepts": len(self.concepts.entries),
            "strategies": len(self.proofs.strategies),
            "arguments": len(self.proofs.arguments),
            "conjectures": len(self.research.conjectures),
            "patterns": len(self.research.patterns),
            "insights": len(self.research.insights),
            "failed_approaches": len(self.research.failed_approaches),
        }
