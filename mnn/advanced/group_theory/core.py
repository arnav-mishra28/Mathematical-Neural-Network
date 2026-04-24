"""
mnn.advanced.group_theory.core
================================
Deep finite group theory: derived series, Sylow theorems,
composition series, homomorphisms, semi-direct products.
"""
from __future__ import annotations
import numpy as np
from itertools import combinations, product as iprod
from typing import List, Dict, Callable, Optional
from mnn.algebra.groups import Group


class FiniteGroupAnalyzer:
    """Extended structural analysis for finite groups."""

    def __init__(self, G: Group):
        self.G = G

    # ── Commutators ───────────────────────────────────────────────────────────
    def commutator(self, a, b):
        """[a, b] = a⁻¹ b⁻¹ a b"""
        G = self.G
        return G.operation(G.operation(G.operation(G.inverse(a), G.inverse(b)), a), b)

    def _generate(self, gens: List) -> List:
        """Closure of a generator set under the group operation."""
        S = set(map(str, gens))
        elems = list(gens)
        for g in gens:
            if self.G.inverse(g) not in [x for x in elems if str(x) in S]:
                iv = self.G.inverse(g)
                if str(iv) not in S:
                    S.add(str(iv)); elems.append(iv)
        changed = True
        while changed:
            changed = False
            new = []
            for a in elems:
                for b in elems:
                    ab = self.G.operation(a, b)
                    if str(ab) not in S:
                        S.add(str(ab)); elems.append(ab); new.append(ab); changed = True
        return elems

    def commutator_subgroup(self, elems: Optional[List] = None) -> List:
        """G' = [G,G] — derived (commutator) subgroup."""
        G = self.G
        elems = elems or G.elements
        e = G.identity()
        gens = [e]
        for a in elems:
            for b in elems:
                c = self.commutator(a, b)
                if str(c) not in [str(x) for x in gens]:
                    gens.append(c)
        return self._generate(gens)

    def derived_series(self) -> List[List]:
        """G⁽⁰⁾ ⊇ G⁽¹⁾ ⊇ ... stopping when stable."""
        series = [self.G.elements]
        for _ in range(self.G.order):
            curr  = series[-1]
            deriv = self.commutator_subgroup(curr)
            if len(deriv) >= len(curr):
                break
            series.append(deriv)
            if len(deriv) <= 1:
                break
        return series

    def is_solvable(self) -> bool:
        """G is solvable iff derived series terminates at {e}."""
        series = self.derived_series()
        return len(series[-1]) == 1

    def lower_central_series(self) -> List[List]:
        """γ₁=G, γₙ₊₁=[γₙ,G]."""
        series = [self.G.elements]
        for _ in range(self.G.order):
            curr = series[-1]
            e    = self.G.identity()
            gens = [e]
            for a in curr:
                for b in self.G.elements:
                    c = self.commutator(a, b)
                    if str(c) not in [str(x) for x in gens]:
                        gens.append(c)
            nxt = self._generate(gens)
            if len(nxt) >= len(curr):
                break
            series.append(nxt)
            if len(nxt) <= 1:
                break
        return series

    def is_nilpotent(self) -> bool:
        return len(self.lower_central_series()[-1]) == 1

    def nilpotency_class(self) -> Optional[int]:
        if not self.is_nilpotent(): return None
        return len(self.lower_central_series()) - 1

    # ── Sylow theorems ────────────────────────────────────────────────────────
    def _subgroups_of_order(self, order: int) -> List[List]:
        G = self.G; e = G.identity(); result = []; seen = []
        rest = [x for x in G.elements if x != e]
        for combo in combinations(rest, order - 1):
            sub = list(combo) + [e]
            if G._is_subgroup(set(sub)):
                key = tuple(sorted(str(x) for x in sub))
                if key not in seen:
                    result.append(sub); seen.append(key)
        return result

    def sylow_subgroups(self, p: int) -> List[List]:
        """All Sylow p-subgroups."""
        n = self.G.order; k = 0
        while n % (p**(k+1)) == 0: k += 1
        return self._subgroups_of_order(p**k) if k > 0 else []

    def sylow_verification(self, p: int) -> Dict:
        """Verify all three Sylow theorems for prime p."""
        n = self.G.order; k = 0
        while n % (p**(k+1)) == 0: k += 1
        sylows = self.sylow_subgroups(p); np_count = len(sylows)
        m = n // max(p**k, 1)
        return {
            "prime": p, "sylow_order": p**k,
            "n_p": np_count,
            "thm1_exists":  np_count > 0 if n % p == 0 else True,
            "thm3_mod_p":   np_count % p == 1 if np_count > 0 else True,
            "thm3_divides": m % np_count == 0 if np_count > 0 else True,
        }

    # ── Composition series ────────────────────────────────────────────────────
    def composition_series(self) -> List[List]:
        """G = G₀ ⊃ G₁ ⊃ ... ⊃ {e} with simple quotients."""
        G = self.G; series = [G.elements]; cur = G
        for _ in range(G.order):
            if len(cur.elements) <= 1: break
            norms = cur.normal_subgroups()
            best  = None; best_ord = 1
            for sub in norms:
                if 1 < sub.order < cur.order and sub.order > best_ord:
                    best = sub; best_ord = sub.order
            if best is None: break
            series.append(best.elements); cur = best
        if series[-1] != [G.identity()]: series.append([G.identity()])
        return series

    def composition_factors(self) -> List[str]:
        s = self.composition_series()
        return [f"C_{len(s[i])//len(s[i+1])}" for i in range(len(s)-1)]

    # ── Structural report ─────────────────────────────────────────────────────
    def structural_report(self) -> str:
        G = self.G
        w = 38
        lines = [
            "╔" + "═"*w + "╗",
            f"║  Group: {G.name:<{w-9}}║",
            "╠" + "═"*w + "╣",
            f"║  Order         : {G.order:<{w-18}}║",
            f"║  Abelian       : {str(G.is_abelian()):<{w-18}}║",
            f"║  Solvable      : {str(self.is_solvable()):<{w-18}}║",
            f"║  Nilpotent     : {str(self.is_nilpotent()):<{w-18}}║",
            f"║  Simple        : {str(G.is_simple()):<{w-18}}║",
            f"║  |Center|      : {len(G.center()):<{w-18}}║",
            f"║  Conj. classes : {len(G.conjugacy_classes()):<{w-18}}║",
            f"║  Derived len   : {len(self.derived_series()):<{w-18}}║",
            f"║  Comp. factors : {str(self.composition_factors()):<{w-18}}║",
            "╚" + "═"*w + "╝",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return f"FiniteGroupAnalyzer({self.G.name}, |G|={self.G.order})"


class GroupHomomorphism:
    """
    φ: G → H. Verify homomorphism property, compute kernel, image,
    test injectivity/surjectivity, apply first isomorphism theorem.
    """
    def __init__(self, G: Group, H: Group, phi: Callable, name="φ"):
        self.G=G; self.H=H; self.phi=phi; self.name=name

    def is_homomorphism(self) -> bool:
        for a in self.G.elements:
            for b in self.G.elements:
                if self.phi(self.G.operation(a,b)) != self.H.operation(self.phi(a),self.phi(b)):
                    return False
        return True

    def kernel(self) -> List:
        eH = self.H.identity()
        return [g for g in self.G.elements if self.phi(g)==eH]

    def image(self) -> List:
        seen=[]; seen_s=set()
        for g in self.G.elements:
            h=self.phi(g); k=str(h)
            if k not in seen_s: seen.append(h); seen_s.add(k)
        return seen

    def is_injective(self) -> bool: return len(self.kernel())==1
    def is_surjective(self) -> bool: return len(self.image())==self.H.order
    def is_isomorphism(self) -> bool: return self.is_homomorphism() and self.is_injective() and self.is_surjective()

    def first_isomorphism_theorem(self) -> str:
        k=len(self.kernel()); q=self.G.order//k; im=len(self.image())
        return f"G/ker(φ) ≅ im(φ)  ↔  |G|/|ker|={q} = |im|={im}  ✓" if q==im else f"Order mismatch: |G|/|ker|={q} ≠ |im|={im}"

    def __repr__(self):
        return f"GroupHom({self.name}: {self.G.name}→{self.H.name}, iso={self.is_isomorphism()})"


class GroupExtension:
    """Semi-direct products and wreath products."""

    @staticmethod
    def semidirect_product(N: Group, Q: Group, action: Callable) -> Group:
        """N ⋊_φ Q where action(q, n) = φ(q)(n)."""
        elems = list(iprod(N.elements, Q.elements))
        def op(a, b):
            n1,q1=a; n2,q2=b
            return (N.operation(n1, action(q1,n2)), Q.operation(q1,q2))
        return Group(elems, op, name=f"{N.name}⋊{Q.name}")

    @staticmethod
    def direct_product(G: Group, H: Group) -> Group:
        return G.direct_product(H)

    def __repr__(self): return "GroupExtension()"
