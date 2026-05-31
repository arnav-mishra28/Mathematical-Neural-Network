"""mnn.mathos.kernel — Mathematical Kernel (Layer 1) + Memory System (Layer 2).

Layer 1: Mathematical Object Model — every entity is a first-class math object.
Natively supports scalars, vectors, tensors, manifolds, groups, categories, PDEs.

Layer 2: Mathematical Memory — accumulates mathematical experience.
Concept, Proof, Discovery, Research, and Agent memory systems.
"""
from __future__ import annotations
import numpy as np
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============= Layer 1: Mathematical Kernel =============

class MathType(Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR = "tensor"
    MATRIX = "matrix"
    MANIFOLD = "manifold"
    GROUP = "group"
    RING = "ring"
    FIELD = "field"
    CATEGORY = "category"
    PDE = "pde"
    OPERATOR = "operator"
    FUNCTION = "function"
    GRAPH = "graph"
    THEOREM = "theorem"
    PROOF = "proof"
    SPACE = "space"


@dataclass
class MathEntity:
    """First-class mathematical object — the universal type in MathOS."""
    name: str
    math_type: MathType
    data: Any = None
    properties: Dict[str, Any] = field(default_factory=dict)
    computed: Dict[str, Any] = field(default_factory=dict)
    relations: List[str] = field(default_factory=list)
    domain: str = ""
    uid: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.uid:
            self.uid = f"{self.math_type.value}_{self.name}_{int(self.timestamp * 1000) % 100000}"

    def info(self) -> str:
        lines = [f"MathEntity({self.name}, {self.math_type.value})"]
        if self.domain:
            lines[0] += f" [{self.domain}]"
        for k, v in self.properties.items():
            lines.append(f"  {k}: {v}")
        for k, v in self.computed.items():
            lines.append(f"  [computed] {k}: {v}")
        return "\n".join(lines)


class MathKernel:
    """Layer 1: The heart of MathOS — manages all mathematical objects."""

    def __init__(self):
        self.entities: Dict[str, MathEntity] = {}
        self.type_registry: Dict[MathType, List[str]] = {t: [] for t in MathType}
        self._auto_compute: Dict[MathType, Callable] = {}
        self._register_auto()
        self.log: List[Dict] = []

    def _register_auto(self):
        self._auto_compute[MathType.SCALAR] = self._auto_scalar
        self._auto_compute[MathType.VECTOR] = self._auto_vector
        self._auto_compute[MathType.MATRIX] = self._auto_matrix
        self._auto_compute[MathType.GROUP] = self._auto_group
        self._auto_compute[MathType.MANIFOLD] = self._auto_manifold

    def create(self, name: str, math_type: str, data: Any = None,
               domain: str = "", **props) -> MathEntity:
        mt = MathType(math_type) if math_type in [e.value for e in MathType] else MathType.SCALAR
        entity = MathEntity(name, mt, data, props, domain=domain)
        auto = self._auto_compute.get(mt)
        if auto:
            auto(entity)
        self.entities[name] = entity
        self.type_registry[mt].append(name)
        self.log.append({"action": "create", "name": name, "type": mt.value})
        return entity

    def get(self, name: str) -> Optional[MathEntity]:
        return self.entities.get(name)

    def delete(self, name: str) -> bool:
        e = self.entities.pop(name, None)
        if e:
            self.type_registry[e.math_type].remove(name)
            return True
        return False

    def by_type(self, math_type: str) -> List[MathEntity]:
        mt = MathType(math_type)
        return [self.entities[n] for n in self.type_registry.get(mt, [])
                if n in self.entities]

    def search(self, query: str) -> List[MathEntity]:
        q = query.lower()
        return [e for e in self.entities.values()
                if q in e.name.lower() or q in e.domain.lower()
                or q in e.math_type.value]

    def relate(self, a: str, b: str, relation: str):
        for n in [a, b]:
            if n in self.entities:
                other = b if n == a else a
                self.entities[n].relations.append(f"{relation}:{other}")

    def _auto_scalar(self, e: MathEntity):
        if isinstance(e.data, (int, float)):
            e.computed["value"] = e.data
            e.computed["is_zero"] = e.data == 0
            e.computed["is_positive"] = e.data > 0

    def _auto_vector(self, e: MathEntity):
        if isinstance(e.data, np.ndarray):
            e.computed["dimension"] = len(e.data)
            e.computed["norm"] = float(np.linalg.norm(e.data))
            e.computed["is_unit"] = abs(np.linalg.norm(e.data) - 1.0) < 1e-10

    def _auto_matrix(self, e: MathEntity):
        if isinstance(e.data, np.ndarray) and e.data.ndim == 2:
            e.computed["shape"] = e.data.shape
            if e.data.shape[0] == e.data.shape[1]:
                e.computed["determinant"] = float(np.linalg.det(e.data))
                e.computed["trace"] = float(np.trace(e.data))
                eigenvalues = np.linalg.eigvals(e.data)
                e.computed["eigenvalues"] = eigenvalues.tolist()
                e.computed["is_symmetric"] = bool(np.allclose(e.data, e.data.T))

    def _auto_group(self, e: MathEntity):
        order = e.properties.get("order")
        if order and isinstance(order, int):
            e.computed["order"] = order
            e.computed["is_finite"] = True
            factors = []
            n = order
            for p in [2, 3, 5, 7, 11, 13, 17, 19]:
                while n % p == 0:
                    factors.append(p)
                    n //= p
                if n == 1:
                    break
            if n > 1:
                factors.append(n)
            e.computed["prime_factorization"] = factors

    def _auto_manifold(self, e: MathEntity):
        dim = e.properties.get("dimension")
        if dim:
            e.computed["dimension"] = dim
        curvature = e.properties.get("curvature")
        if curvature is not None:
            e.computed["curvature_type"] = (
                "flat" if curvature == 0 else "positive" if curvature > 0 else "negative")
        radius = e.properties.get("radius")
        if radius and dim == 2:
            e.computed["area"] = 4 * np.pi * radius ** 2
            e.computed["euler_characteristic"] = 2

    def summary(self) -> str:
        counts = {t.value: len(names) for t, names in self.type_registry.items() if names}
        return f"MathKernel: {len(self.entities)} entities, types={counts}"


# ============= Layer 2: Mathematical Memory =============

class MemoryKind(Enum):
    CONCEPT = "concept"
    PROOF = "proof"
    DISCOVERY = "discovery"
    RESEARCH = "research"
    AGENT = "agent"


@dataclass
class MemoryRecord:
    """A record in mathematical memory."""
    kind: MemoryKind
    key: str
    content: str
    importance: float = 1.0
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def access(self):
        self.access_count += 1


class MathMemorySystem:
    """Layer 2: Unified mathematical memory — accumulates experience."""

    def __init__(self):
        self.records: Dict[str, MemoryRecord] = {}

    def store(self, kind: str, key: str, content: str,
              importance: float = 1.0, tags: List[str] = None):
        mk = MemoryKind(kind) if kind in [e.value for e in MemoryKind] else MemoryKind.CONCEPT
        self.records[key] = MemoryRecord(mk, key, content, importance, tags=tags or [])

    def recall(self, key: str) -> Optional[str]:
        rec = self.records.get(key)
        if rec:
            rec.access()
            return rec.content
        return None

    def search(self, query: str) -> List[MemoryRecord]:
        q = query.lower()
        return [r for r in self.records.values()
                if q in r.key.lower() or q in r.content.lower()
                or any(q in t for t in r.tags)]

    def by_kind(self, kind: str) -> List[MemoryRecord]:
        mk = MemoryKind(kind)
        return [r for r in self.records.values() if r.kind == mk]

    def most_important(self, top_k: int = 10) -> List[MemoryRecord]:
        return sorted(self.records.values(), key=lambda r: -r.importance)[:top_k]

    def most_accessed(self, top_k: int = 10) -> List[MemoryRecord]:
        return sorted(self.records.values(), key=lambda r: -r.access_count)[:top_k]

    def summary(self) -> str:
        counts = {}
        for r in self.records.values():
            counts[r.kind.value] = counts.get(r.kind.value, 0) + 1
        return f"MathMemory: {len(self.records)} records, kinds={counts}"
