"""mnn.embeddings.theorem_states — Theorem State Representation.

Part 1: Represent theorems as quantum states |T> = sum_i c_i |b_i>
where b_i are basis mathematical concepts and c_i are complex amplitudes.
A theorem becomes a geometric probability structure in Hilbert space.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import hashlib


@dataclass
class MathConcept:
    """A basis mathematical concept (an axis in theorem space)."""
    name: str
    category: str  # "algebra", "analysis", "geometry", "topology", "logic", etc.
    level: int = 0  # abstraction level (0=primitive, higher=more abstract)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(f"{self.name}:{self.category}")

    def __eq__(self, other):
        return isinstance(other, MathConcept) and self.name == other.name and self.category == other.category


class ConceptBasis:
    """A basis of mathematical concepts spanning theorem space.

    Each concept is an orthonormal basis vector |b_i> in the Hilbert space.
    """
    # Standard mathematical concept vocabulary
    STANDARD_CONCEPTS = [
        MathConcept("commutativity", "algebra"),
        MathConcept("associativity", "algebra"),
        MathConcept("identity_element", "algebra"),
        MathConcept("inverse", "algebra"),
        MathConcept("distributivity", "algebra"),
        MathConcept("closure", "algebra"),
        MathConcept("homomorphism", "algebra"),
        MathConcept("isomorphism", "algebra"),
        MathConcept("continuity", "analysis"),
        MathConcept("differentiability", "analysis"),
        MathConcept("integrability", "analysis"),
        MathConcept("convergence", "analysis"),
        MathConcept("completeness", "analysis"),
        MathConcept("compactness", "topology"),
        MathConcept("connectedness", "topology"),
        MathConcept("homeomorphism", "topology"),
        MathConcept("metric", "geometry"),
        MathConcept("curvature", "geometry"),
        MathConcept("geodesic", "geometry"),
        MathConcept("symmetry", "geometry"),
        MathConcept("linearity", "linear_algebra"),
        MathConcept("eigenvalue", "linear_algebra"),
        MathConcept("orthogonality", "linear_algebra"),
        MathConcept("projection", "linear_algebra"),
        MathConcept("implication", "logic"),
        MathConcept("equivalence", "logic"),
        MathConcept("quantification", "logic"),
        MathConcept("negation", "logic"),
        MathConcept("induction", "logic"),
        MathConcept("conservation", "dynamics"),
        MathConcept("stability", "dynamics"),
        MathConcept("periodicity", "dynamics"),
    ]

    def __init__(self, concepts: Optional[List[MathConcept]] = None):
        self.concepts = concepts or self.STANDARD_CONCEPTS.copy()
        self.dim = len(self.concepts)
        self._index = {c.name: i for i, c in enumerate(self.concepts)}

    def index(self, name: str) -> int:
        return self._index[name]

    def add_concept(self, concept: MathConcept):
        if concept.name not in self._index:
            self._index[concept.name] = self.dim
            self.concepts.append(concept)
            self.dim += 1

    def get_category_mask(self, category: str) -> np.ndarray:
        """Boolean mask for concepts in a given category."""
        return np.array([c.category == category for c in self.concepts])

    def __repr__(self):
        cats = {}
        for c in self.concepts:
            cats.setdefault(c.category, []).append(c.name)
        lines = [f"ConceptBasis(dim={self.dim})"]
        for cat, names in sorted(cats.items()):
            lines.append(f"  {cat}: {', '.join(names)}")
        return "\n".join(lines)


class TheoremState:
    """A theorem represented as a quantum state |T> in concept Hilbert space.

    |T> = sum_i c_i |b_i> where c_i are complex amplitudes encoding
    the theorem's relationship to each mathematical concept.
    """
    def __init__(self, amplitudes: np.ndarray, basis: ConceptBasis,
                 name: str = "", statement: str = "",
                 metadata: Optional[Dict] = None):
        self.amplitudes = np.asarray(amplitudes, dtype=complex)
        self.basis = basis
        self.name = name
        self.statement = statement
        self.metadata = metadata or {}
        assert len(self.amplitudes) == basis.dim, \
            f"Amplitude dim {len(self.amplitudes)} != basis dim {basis.dim}"
        self._normalize()

    def _normalize(self):
        norm = np.linalg.norm(self.amplitudes)
        if norm > 1e-15:
            self.amplitudes /= norm

    @classmethod
    def from_concepts(cls, concept_weights: Dict[str, complex],
                      basis: ConceptBasis, name: str = "",
                      statement: str = "") -> "TheoremState":
        """Create from concept name -> amplitude mapping."""
        amps = np.zeros(basis.dim, dtype=complex)
        for cname, weight in concept_weights.items():
            if cname in basis._index:
                amps[basis.index(cname)] = weight
        return cls(amps, basis, name, statement)

    @classmethod
    def from_keywords(cls, keywords: List[str], basis: ConceptBasis,
                      name: str = "", statement: str = "") -> "TheoremState":
        """Create from keyword list (equal real amplitudes)."""
        weights = {kw: 1.0 for kw in keywords if kw in basis._index}
        return cls.from_concepts(weights, basis, name, statement)

    @classmethod
    def random(cls, basis: ConceptBasis, name: str = "",
               seed: Optional[int] = None) -> "TheoremState":
        rng = np.random.default_rng(seed)
        real = rng.standard_normal(basis.dim)
        imag = rng.standard_normal(basis.dim)
        return cls(real + 1j * imag, basis, name)

    def inner_product(self, other: "TheoremState") -> complex:
        """<self|other>"""
        return complex(np.vdot(self.amplitudes, other.amplitudes))

    def similarity(self, other: "TheoremState") -> float:
        """|<self|other>|^2 — quantum overlap probability."""
        return abs(self.inner_product(other)) ** 2

    def fidelity(self, other: "TheoremState") -> float:
        """|<self|other>| — state fidelity."""
        return abs(self.inner_product(other))

    def probabilities(self) -> np.ndarray:
        """Probability of each concept."""
        return np.abs(self.amplitudes) ** 2

    def top_concepts(self, n: int = 5) -> List[Tuple[str, float]]:
        """Top n concepts by probability."""
        probs = self.probabilities()
        top_idx = np.argsort(-probs)[:n]
        return [(self.basis.concepts[i].name, float(probs[i])) for i in top_idx]

    def density_matrix(self) -> np.ndarray:
        psi = self.amplitudes.reshape(-1, 1)
        return psi @ psi.conj().T

    def concept_entropy(self) -> float:
        """Shannon entropy of concept distribution."""
        probs = self.probabilities()
        probs = probs[probs > 1e-15]
        return float(-np.sum(probs * np.log2(probs)))

    def category_projection(self, category: str) -> "TheoremState":
        """Project onto a mathematical category subspace."""
        mask = self.basis.get_category_mask(category)
        new_amps = self.amplitudes * mask
        return TheoremState(new_amps, self.basis, f"{self.name}|_{category}")

    def superpose(self, other: "TheoremState", alpha: complex = 0.5+0j,
                  beta: complex = 0.5+0j) -> "TheoremState":
        """Superposition: alpha|self> + beta|other>."""
        amps = alpha * self.amplitudes + beta * other.amplitudes
        return TheoremState(amps, self.basis,
                           f"({self.name}+{other.name})")

    def evolve(self, unitary: np.ndarray) -> "TheoremState":
        """Apply unitary transformation."""
        return TheoremState(unitary @ self.amplitudes, self.basis, self.name)

    def fingerprint(self) -> str:
        h = hashlib.md5(self.amplitudes.tobytes()).hexdigest()[:12]
        return f"{self.name}:{h}"

    def __repr__(self):
        top = self.top_concepts(3)
        top_str = ", ".join(f"{n}={p:.2f}" for n, p in top)
        return f"TheoremState({self.name}, dim={self.basis.dim}, [{top_str}])"
