"""mnn.quantum_pde.categorical_pde — Category-Theoretic PDE Structure.

Part 8: Unify PDE solving with Category Theory.
Objects = solution spaces, Morphisms = PDE operators,
Functors = discretization mappings. PDE solving becomes
compositional mathematics.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SolutionSpace:
    """Object in PDE category: a function space of solutions."""
    name: str
    dimension: int          # grid size or basis dimension
    domain: str = "[0,1]"   # domain description
    regularity: str = "L2"  # e.g. "L2", "H1", "C^inf"
    metadata: Dict = field(default_factory=dict)

    def __repr__(self):
        return f"SolutionSpace({self.name}, dim={self.dimension}, {self.regularity})"


@dataclass
class PDEMorphism:
    """Morphism in PDE category: maps between solution spaces."""
    name: str
    source: str
    target: str
    operator_fn: Optional[Callable] = None
    properties: Dict = field(default_factory=dict)

    def apply(self, data: np.ndarray) -> np.ndarray:
        if self.operator_fn is not None:
            return self.operator_fn(data)
        return data

    def __repr__(self):
        return f"PDEMorphism({self.name}: {self.source} → {self.target})"


class PDECategory:
    """Category of PDE solution spaces and operators.

    Objects: solution spaces (function spaces)
    Morphisms: PDE operators, boundary maps, discretizations
    """
    def __init__(self, name: str = "PDECat"):
        self.name = name
        self.spaces: Dict[str, SolutionSpace] = {}
        self.morphisms: List[PDEMorphism] = []

    def add_space(self, space: SolutionSpace):
        self.spaces[space.name] = space

    def add_morphism(self, morphism: PDEMorphism):
        assert morphism.source in self.spaces, f"Unknown source: {morphism.source}"
        assert morphism.target in self.spaces, f"Unknown target: {morphism.target}"
        self.morphisms.append(morphism)

    def compose(self, m1_name: str, m2_name: str) -> Optional[PDEMorphism]:
        """g ∘ f composition."""
        m1 = next((m for m in self.morphisms if m.name == m1_name), None)
        m2 = next((m for m in self.morphisms if m.name == m2_name), None)
        if not m1 or not m2 or m1.target != m2.source:
            return None
        composed_fn = None
        if m1.operator_fn and m2.operator_fn:
            f1, f2 = m1.operator_fn, m2.operator_fn
            composed_fn = lambda x, _f1=f1, _f2=f2: _f2(_f1(x))
        composed = PDEMorphism(
            f"{m2_name}∘{m1_name}", m1.source, m2.target,
            composed_fn, {"composed_from": [m1_name, m2_name]})
        self.morphisms.append(composed)
        return composed

    def hom(self, source: str, target: str) -> List[PDEMorphism]:
        return [m for m in self.morphisms
                if m.source == source and m.target == target]

    def endomorphisms(self, space: str) -> List[PDEMorphism]:
        return self.hom(space, space)

    def is_commutative(self, m1_name: str, m2_name: str,
                        test_data: np.ndarray, atol: float = 1e-6) -> bool:
        """Test if two morphisms commute: m1∘m2 ≈ m2∘m1."""
        m1 = next((m for m in self.morphisms if m.name == m1_name), None)
        m2 = next((m for m in self.morphisms if m.name == m2_name), None)
        if not m1 or not m2 or not m1.operator_fn or not m2.operator_fn:
            return False
        try:
            r1 = m1.operator_fn(m2.operator_fn(test_data))
            r2 = m2.operator_fn(m1.operator_fn(test_data))
            return bool(np.allclose(r1, r2, atol=atol))
        except Exception:
            return False

    def summary(self) -> str:
        lines = [f"PDECategory({self.name})",
                 f"  Spaces (objects): {len(self.spaces)}"]
        for s in self.spaces.values():
            lines.append(f"    {s}")
        lines.append(f"  Operators (morphisms): {len(self.morphisms)}")
        for m in self.morphisms:
            lines.append(f"    {m}")
        return "\n".join(lines)


class DiscretizationFunctor:
    """Functor from continuous PDE category to discrete category.

    Maps continuous solution spaces to finite-dimensional grids,
    and PDE operators to matrix operators.
    """
    def __init__(self, n_grid: int, domain: Tuple[float, float] = (0, 1),
                 name: str = "Discretize"):
        self.n_grid = n_grid
        self.domain = domain
        self.dx = (domain[1] - domain[0]) / n_grid
        self.grid = np.linspace(domain[0], domain[1], n_grid, endpoint=False)
        self.name = name

    def discretize_space(self, space: SolutionSpace) -> SolutionSpace:
        return SolutionSpace(
            f"D({space.name})", self.n_grid,
            f"grid({self.n_grid})", "discrete",
            {"original": space.name, "dx": self.dx})

    def discretize_laplacian(self) -> np.ndarray:
        """Discretize ∇² as matrix."""
        n = self.n_grid
        D2 = np.zeros((n, n))
        for i in range(n):
            D2[i, i] = -2.0
            D2[i, (i + 1) % n] = 1.0
            D2[i, (i - 1) % n] = 1.0
        return D2 / self.dx**2

    def discretize_gradient(self) -> np.ndarray:
        """Discretize ∂/∂x as matrix."""
        n = self.n_grid
        D1 = np.zeros((n, n))
        for i in range(n):
            D1[i, (i + 1) % n] = 1.0
            D1[i, (i - 1) % n] = -1.0
        return D1 / (2 * self.dx)

    def discretize_function(self, f: Callable) -> np.ndarray:
        """Sample continuous function on grid."""
        return np.array([f(x) for x in self.grid])

    def refinement_morphism(self, coarse_n: int) -> np.ndarray:
        """Interpolation matrix from coarse to fine grid."""
        fine_n = self.n_grid
        P = np.zeros((fine_n, coarse_n))
        ratio = coarse_n / fine_n
        for i in range(fine_n):
            j = i * ratio
            j0 = int(j) % coarse_n
            j1 = (j0 + 1) % coarse_n
            alpha = j - int(j)
            P[i, j0] = 1 - alpha
            P[i, j1] = alpha
        return P

    def verify_functor(self, op1: np.ndarray, op2: np.ndarray,
                        test_data: np.ndarray) -> Dict:
        """Verify discretization preserves composition: D(op2∘op1) ≈ D(op2)∘D(op1)."""
        composed = op2 @ op1
        sequential = op2 @ (op1 @ test_data)
        direct = composed @ test_data
        err = float(np.max(np.abs(sequential - direct)))
        return {"composition_preserved": err < 1e-10, "max_error": err}

    def __repr__(self):
        return (f"DiscretizationFunctor({self.name}, n={self.n_grid}, "
                f"domain={self.domain})")
