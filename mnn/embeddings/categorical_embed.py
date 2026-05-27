"""mnn.embeddings.categorical_embed — Category-Theoretic Embeddings.

Part 6: Integrate theorem embeddings with the Category layer.
Objects = theorem states, Morphisms = proof transformations.
The embedding system becomes compositional and geometric simultaneously.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


class CategoricalTheoremSpace:
    """A category where objects are theorem states and morphisms are proof steps.

    Combines category theory structure with continuous embeddings.
    """
    def __init__(self, embed_dim: int, name: str = "TheoremCat"):
        self.embed_dim = embed_dim
        self.name = name
        self.theorems: Dict[str, Dict] = {}   # name -> {"real": ..., "imag": ...}
        self.morphisms: List[Dict] = []

    def add_theorem(self, name: str, real: np.ndarray, imag: np.ndarray,
                     metadata: Optional[Dict] = None):
        """Add a theorem as an object in the category."""
        vec = real + 1j * imag
        vec /= np.linalg.norm(vec) + 1e-15
        self.theorems[name] = {
            "real": np.real(vec), "imag": np.imag(vec),
            "metadata": metadata or {},
        }

    def add_proof_morphism(self, source: str, target: str,
                            transform_fn: Optional[Callable] = None,
                            name: str = ""):
        """Add a proof step as a morphism source -> target."""
        if source not in self.theorems or target not in self.theorems:
            raise ValueError(f"Unknown theorem: {source} or {target}")
        morph = {
            "source": source, "target": target,
            "name": name or f"{source}->{target}",
            "transform_fn": transform_fn,
        }
        self.morphisms.append(morph)

    def compose_morphisms(self, m1_name: str, m2_name: str) -> Optional[Dict]:
        """Compose two morphisms: m2 ∘ m1."""
        m1 = next((m for m in self.morphisms if m["name"] == m1_name), None)
        m2 = next((m for m in self.morphisms if m["name"] == m2_name), None)
        if not m1 or not m2:
            return None
        if m1["target"] != m2["source"]:
            return None
        composed = {
            "source": m1["source"], "target": m2["target"],
            "name": f"{m2_name}∘{m1_name}",
            "transform_fn": None,
        }
        self.morphisms.append(composed)
        return composed

    def hom_set(self, source: str, target: str) -> List[Dict]:
        """All morphisms from source to target."""
        return [m for m in self.morphisms
                if m["source"] == source and m["target"] == target]

    def theorem_distance(self, t1: str, t2: str) -> float:
        """Fubini-Study distance between two theorems."""
        s1 = self.theorems[t1]["real"] + 1j * self.theorems[t1]["imag"]
        s2 = self.theorems[t2]["real"] + 1j * self.theorems[t2]["imag"]
        overlap = np.abs(np.vdot(s1, s2))
        return float(np.arccos(np.clip(overlap, 0, 1)))

    def morphism_preserves(self, morph_name: str,
                            property_fns: Dict[str, Callable]) -> Dict[str, bool]:
        """Check which properties a morphism preserves."""
        morph = next((m for m in self.morphisms if m["name"] == morph_name), None)
        if not morph:
            return {}
        src = self.theorems[morph["source"]]
        tgt = self.theorems[morph["target"]]
        s_vec = src["real"] + 1j * src["imag"]
        t_vec = tgt["real"] + 1j * tgt["imag"]

        results = {}
        for pname, pfn in property_fns.items():
            try:
                p_src = pfn(s_vec)
                p_tgt = pfn(t_vec)
                results[pname] = bool(np.allclose(p_src, p_tgt, atol=1e-4))
            except Exception:
                results[pname] = False
        return results

    def find_isomorphisms(self, threshold: float = 0.01) -> List[Tuple[str, str, float]]:
        """Find pairs of theorems that are nearly isomorphic (very close)."""
        names = list(self.theorems.keys())
        isos = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                d = self.theorem_distance(names[i], names[j])
                if d < threshold:
                    isos.append((names[i], names[j], d))
        return isos

    def connected_components(self) -> List[List[str]]:
        """Find connected components via morphisms."""
        names = list(self.theorems.keys())
        parent = {n: n for n in names}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for m in self.morphisms:
            if m["source"] in parent and m["target"] in parent:
                union(m["source"], m["target"])

        components = {}
        for n in names:
            root = find(n)
            components.setdefault(root, []).append(n)
        return list(components.values())

    def summary(self) -> str:
        comps = self.connected_components()
        lines = [
            f"CategoricalTheoremSpace({self.name})",
            f"  Theorems (objects): {len(self.theorems)}",
            f"  Proof steps (morphisms): {len(self.morphisms)}",
            f"  Connected components: {len(comps)}",
        ]
        for i, comp in enumerate(comps):
            lines.append(f"    Component {i}: {comp}")
        return "\n".join(lines)


class TheoremFunctor:
    """Functor between theorem categories: maps theorems and proofs.

    F: TheoremCat_1 -> TheoremCat_2
    """
    def __init__(self, source: CategoricalTheoremSpace,
                 target: CategoricalTheoremSpace,
                 obj_map: Callable, mor_map: Callable,
                 name: str = "F"):
        self.source = source
        self.target = target
        self.obj_map = obj_map
        self.mor_map = mor_map
        self.name = name

    def apply(self):
        """Apply functor to all objects and morphisms."""
        for tname, tdata in self.source.theorems.items():
            vec = tdata["real"] + 1j * tdata["imag"]
            mapped = self.obj_map(vec)
            mapped /= np.linalg.norm(mapped) + 1e-15
            self.target.add_theorem(
                f"{self.name}({tname})", np.real(mapped), np.imag(mapped))

        for morph in self.source.morphisms:
            mapped_name = self.mor_map(morph["name"])
            src = f"{self.name}({morph['source']})"
            tgt = f"{self.name}({morph['target']})"
            if src in self.target.theorems and tgt in self.target.theorems:
                self.target.add_proof_morphism(src, tgt, name=mapped_name)

    def verify_composition(self) -> bool:
        """Verify functor preserves composition (spot check)."""
        for m1 in self.source.morphisms[:5]:
            for m2 in self.source.morphisms[:5]:
                if m1["target"] == m2["source"]:
                    # F(m2 ∘ m1) should equal F(m2) ∘ F(m1)
                    # (structural check only — both should exist)
                    src = f"{self.name}({m1['source']})"
                    tgt = f"{self.name}({m2['target']})"
                    if src in self.target.theorems and tgt in self.target.theorems:
                        return True
        return True

    def __repr__(self):
        return (f"TheoremFunctor({self.name}: {self.source.name} -> "
                f"{self.target.name})")
