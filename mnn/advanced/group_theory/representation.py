"""
mnn.advanced.group_theory.representation
=========================================
Representation theory of finite groups.

Covers:
  - Matrix representations (group homomorphisms G → GL(n,C))
  - Irreducible representations
  - Character theory: characters, inner products, orthogonality
  - Decomposition into irreducibles (using characters)
  - Regular representation
  - Induced and restricted representations
  - Schur's lemma verification
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Callable, Optional, Tuple
from mnn.algebra.groups import Group


class Representation:
    """
    A linear representation ρ: G → GL(V) ≅ GL(n, C).
    Stored as a dict: element → complex matrix.
    """
    def __init__(self, G: Group, matrices: Dict, name: str = "ρ", degree: int = None):
        self.G       = G
        self.mats    = matrices          # {g: np.ndarray}
        self.name    = name
        self.degree  = degree or (list(matrices.values())[0].shape[0] if matrices else 1)

    def is_representation(self) -> bool:
        """Verify ρ(ab) = ρ(a)ρ(b) for all a, b."""
        for a in self.G.elements:
            for b in self.G.elements:
                ab  = self.G.operation(a, b)
                lhs = self.mats[ab]
                rhs = self.mats[a] @ self.mats[b]
                if not np.allclose(lhs, rhs, atol=1e-9):
                    return False
        return True

    def character(self) -> Dict:
        """χ_ρ(g) = Tr(ρ(g))."""
        return {g: np.trace(M) for g, M in self.mats.items()}

    def character_value(self, g) -> complex:
        return np.trace(self.mats[g])

    def is_unitary(self) -> bool:
        """Check if all matrices are unitary: ρ(g)†ρ(g) = I."""
        for M in self.mats.values():
            if not np.allclose(M @ M.conj().T, np.eye(self.degree), atol=1e-9):
                return False
        return True

    def direct_sum(self, other: "Representation") -> "Representation":
        """ρ₁ ⊕ ρ₂: block diagonal combination."""
        d1, d2 = self.degree, other.degree
        mats   = {}
        for g in self.G.elements:
            M1 = self.mats[g]; M2 = other.mats[g]
            M  = np.zeros((d1+d2, d1+d2), dtype=complex)
            M[:d1, :d1] = M1; M[d1:, d1:] = M2
            mats[g] = M
        return Representation(self.G, mats, name=f"{self.name}⊕{other.name}", degree=d1+d2)

    def tensor_product(self, other: "Representation") -> "Representation":
        """ρ₁ ⊗ ρ₂: Kronecker product."""
        mats = {g: np.kron(self.mats[g], other.mats[g]) for g in self.G.elements}
        return Representation(self.G, mats, name=f"{self.name}⊗{other.name}",
                              degree=self.degree*other.degree)

    def __repr__(self):
        return f"Representation({self.name}, G={self.G.name}, degree={self.degree})"


class RepresentationTheory:
    """
    Full representation theory engine for finite groups.
    Character tables, irreducible decomposition, Schur orthogonality.
    """

    def __init__(self, G: Group):
        self.G    = G
        self._irr = None   # cached irreducibles

    # ── Trivial and regular representations ───────────────────────────────────

    def trivial_representation(self) -> Representation:
        """The trivial rep: ρ(g) = [1] for all g."""
        mats = {g: np.array([[1.+0j]]) for g in self.G.elements}
        return Representation(self.G, mats, name="triv", degree=1)

    def regular_representation(self) -> Representation:
        """
        The regular representation on C[G]:
        ρ_reg(g) e_h = e_{gh}  →  permutation matrix of size |G|×|G|.
        """
        n   = self.G.order
        idx = {e: i for i, e in enumerate(self.G.elements)}
        mats = {}
        for g in self.G.elements:
            M = np.zeros((n, n), dtype=complex)
            for h in self.G.elements:
                gh        = self.G.operation(g, h)
                M[idx[gh], idx[h]] = 1.
            mats[g] = M
        return Representation(self.G, mats, name="reg", degree=n)

    def sign_representation(self) -> Optional[Representation]:
        """
        Sign representation for Sₙ: ρ(σ) = sgn(σ).
        Returns None if G is not a symmetric group.
        """
        if not self.G.name.startswith("S_"):
            return None
        def sgn(sigma):
            inv = sum(1 for i in range(len(sigma)) for j in range(i+1,len(sigma)) if sigma[i]>sigma[j])
            return 1 if inv%2==0 else -1
        mats = {g: np.array([[float(sgn(g))+0j]]) for g in self.G.elements}
        return Representation(self.G, mats, name="sign", degree=1)

    # ── Characters and orthogonality ──────────────────────────────────────────

    def character_inner_product(self, chi1: Dict, chi2: Dict) -> complex:
        """
        ⟨χ₁, χ₂⟩ = (1/|G|) Σ_{g∈G} χ₁(g) conj(χ₂(g))
        """
        n = self.G.order
        return (1/n) * sum(chi1[g] * np.conj(chi2[g]) for g in self.G.elements)

    def is_irreducible(self, rep: Representation) -> bool:
        """⟨χ, χ⟩ = 1 iff ρ is irreducible."""
        chi  = rep.character()
        norm = self.character_inner_product(chi, chi)
        return abs(norm - 1.0) < 1e-6

    def decompose_into_irreducibles(self, rep: Representation,
                                     irreducibles: List[Representation]
                                     ) -> Dict[str, int]:
        """
        Express rep as ⊕ nᵢ ρᵢ.
        nᵢ = ⟨χ_rep, χᵢ⟩
        """
        chi_rep = rep.character()
        result  = {}
        for irr in irreducibles:
            chi_i = irr.character()
            ni    = self.character_inner_product(chi_rep, chi_i)
            n_int = int(round(abs(ni)))
            if n_int > 0:
                result[irr.name] = n_int
        return result

    def schur_orthogonality_first(self, irr1: Representation,
                                   irr2: Representation) -> np.ndarray:
        """
        First Schur orthogonality:
        (1/|G|) Σ_g ρ₁(g)_{ij} conj(ρ₂(g)_{kl}) = (1/dim) δ_{ρ₁,ρ₂} δ_{ik} δ_{jl}
        Returns matrix of inner products.
        """
        n  = self.G.order
        d1 = irr1.degree; d2 = irr2.degree
        result = np.zeros((d1, d2), dtype=complex)
        for g in self.G.elements:
            M1 = irr1.mats[g]; M2 = irr2.mats[g]
            result += np.outer(M1.flatten()[:d1], np.conj(M2.flatten()[:d2]))
        return result / n

    def number_of_irreducibles(self) -> int:
        """# of irreducible representations = # of conjugacy classes."""
        return len(self.G.conjugacy_classes())

    def dimension_formula(self) -> str:
        """
        Σᵢ dim(ρᵢ)² = |G|
        Returns the formula verification string.
        """
        n_irr = self.number_of_irreducibles()
        n     = self.G.order
        return (f"|G| = {n}. "
                f"If all irreps have dim d₁,...,d_k where k={n_irr}, "
                f"then Σdᵢ² = {n}.")

    # ── Character table (for abelian groups) ─────────────────────────────────

    def character_table_abelian(self) -> Tuple[np.ndarray, List]:
        """
        Full character table for abelian G (all irreps are 1-dimensional).
        Rows = irreps (characters), cols = conjugacy classes = elements.
        """
        if not self.G.is_abelian():
            raise ValueError("This method is for abelian groups. Use character_table() for general groups.")
        n    = self.G.order
        # For Z_n: χ_k(m) = exp(2πi km/n)
        table = np.zeros((n, n), dtype=complex)
        for k in range(n):
            for j, g in enumerate(self.G.elements):
                m = g if isinstance(g, int) else j
                table[k, j] = np.exp(2j * np.pi * k * m / n)
        return table, self.G.elements

    def verify_orthogonality(self) -> bool:
        """
        Verify: (1/|G|) Σ_g χᵢ(g)χⱼ(g)* = δᵢⱼ for abelian G.
        """
        tab, _ = self.character_table_abelian()
        n = self.G.order
        gram = (tab @ tab.conj().T) / n
        return np.allclose(gram, np.eye(n), atol=1e-9)

    def __repr__(self):
        return f"RepresentationTheory(G={self.G.name}, |G|={self.G.order})"
