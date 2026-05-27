"""mnn.discovery.categorical — Category-Theoretic Discovery Engine.

Engine 6: Connects back to the Category Theory layer. Discovers
morphism invariants, compositional structures, conserved properties
under functorial mappings, and equivalence classes.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from .conjectures import Conjecture, ConjectureType, ConjectureGenerator


class MorphismInvariantFinder:
    """Discover properties preserved by morphisms and functors.

    If f: A -> B preserves structure, infer conserved properties
    and equivalence classes.
    """

    @staticmethod
    def find_preserved_properties(morphism_fn: Callable,
                                    property_fns: Dict[str, Callable],
                                    test_data: List[Any],
                                    name: str = "f") -> List[Dict]:
        """Test which properties are preserved: P(f(x)) = P(x).

        property_fns: {name: function} where each function computes a property.
        """
        preserved = []
        for pname, pfn in property_fns.items():
            n_pass = 0
            n_total = 0
            for x in test_data:
                try:
                    px = pfn(x)
                    fx = morphism_fn(x)
                    pfx = pfn(fx)
                    n_total += 1
                    if isinstance(px, np.ndarray):
                        if np.allclose(px, pfx, atol=1e-6):
                            n_pass += 1
                    elif abs(px - pfx) < 1e-6:
                        n_pass += 1
                except Exception:
                    pass

            if n_total > 0 and n_pass / n_total > 0.99:
                preserved.append({
                    "property": pname,
                    "preservation_rate": n_pass / n_total,
                    "n_tested": n_total,
                })
        return preserved

    @staticmethod
    def find_composition_invariants(f: Callable, g: Callable,
                                      property_fns: Dict[str, Callable],
                                      test_data: List[Any]) -> List[Dict]:
        """Find properties preserved by g ∘ f that neither preserves alone."""
        composed = lambda x: g(f(x))
        inv_f = MorphismInvariantFinder.find_preserved_properties(f, property_fns, test_data, "f")
        inv_g = MorphismInvariantFinder.find_preserved_properties(g, property_fns, test_data, "g")
        inv_gf = MorphismInvariantFinder.find_preserved_properties(composed, property_fns, test_data, "g∘f")

        f_names = {d["property"] for d in inv_f}
        g_names = {d["property"] for d in inv_g}
        gf_names = {d["property"] for d in inv_gf}

        emergent = gf_names - (f_names | g_names)
        return [d for d in inv_gf if d["property"] in emergent]


class EquivalenceDiscovery:
    """Discover equivalence classes under morphisms/transformations."""

    @staticmethod
    def find_equivalence_classes(elements: List[Any],
                                   equiv_fn: Callable,
                                   atol: float = 1e-6) -> List[List[int]]:
        """Group elements into equivalence classes.

        equiv_fn(a, b) -> True if a ~ b.
        Returns list of index groups.
        """
        n = len(elements)
        visited = [False] * n
        classes = []

        for i in range(n):
            if visited[i]:
                continue
            cls = [i]
            visited[i] = True
            for j in range(i + 1, n):
                if not visited[j]:
                    try:
                        if equiv_fn(elements[i], elements[j]):
                            cls.append(j)
                            visited[j] = True
                    except Exception:
                        pass
            classes.append(cls)
        return classes

    @staticmethod
    def find_orbit(element: Any, group_actions: List[Callable]) -> List[Any]:
        """Find the orbit of an element under a group of transformations."""
        orbit = [element]
        seen = {str(element)}
        queue = [element]

        while queue:
            current = queue.pop(0)
            for action in group_actions:
                try:
                    transformed = action(current)
                    key = str(transformed)
                    if key not in seen:
                        seen.add(key)
                        orbit.append(transformed)
                        queue.append(transformed)
                except Exception:
                    pass
            if len(orbit) > 1000:  # safety limit
                break

        return orbit

    @staticmethod
    def quotient_structure(elements: List[Any],
                            equiv_fn: Callable,
                            op: Optional[Callable] = None) -> Dict:
        """Compute quotient structure X/~ and check if operation descends."""
        classes = EquivalenceDiscovery.find_equivalence_classes(elements, equiv_fn)
        result = {
            "n_elements": len(elements),
            "n_classes": len(classes),
            "class_sizes": [len(c) for c in classes],
        }

        if op is not None:
            # Check if operation is well-defined on quotient
            well_defined = True
            for cls in classes:
                for i, j in zip(cls[:-1], cls[1:]):
                    for k in range(min(len(elements), 10)):
                        try:
                            r1 = op(elements[i], elements[k])
                            r2 = op(elements[j], elements[k])
                            if not equiv_fn(r1, r2):
                                well_defined = False
                                break
                        except Exception:
                            pass
                if not well_defined:
                    break
            result["op_well_defined_on_quotient"] = well_defined

        return result


class FunctorialDiscovery:
    """Discover functorial relationships between mathematical structures."""

    @staticmethod
    def test_functor_axioms(obj_map: Callable, mor_map: Callable,
                              objects: List[Any],
                              morphisms: List[Tuple[Any, Any, Callable]],
                              test_data: List[Any]) -> Dict:
        """Test if (obj_map, mor_map) satisfies functor axioms.

        morphisms: list of (domain, codomain, function) triples.
        """
        results = {"identity_preservation": True, "composition_preservation": True}

        # Identity preservation: F(id_A) = id_{F(A)}
        for obj in objects:
            try:
                FA = obj_map(obj)
                id_mapped = mor_map(lambda x: x)  # identity morphism
                for x in test_data[:5]:
                    try:
                        if not np.allclose(id_mapped(x), x, atol=1e-6):
                            results["identity_preservation"] = False
                    except Exception:
                        pass
            except Exception:
                pass

        # Composition preservation: F(g∘f) = F(g)∘F(f)
        for i, (dA, cA, f) in enumerate(morphisms):
            for j, (dB, cB, g) in enumerate(morphisms):
                if str(cA) == str(dB):
                    try:
                        gf = lambda x, _f=f, _g=g: _g(_f(x))
                        F_gf = mor_map(gf)
                        Ff = mor_map(f)
                        Fg = mor_map(g)
                        Fg_Ff = lambda x, _Ff=Ff, _Fg=Fg: _Fg(_Ff(x))
                        for x in test_data[:3]:
                            try:
                                r1 = F_gf(x)
                                r2 = Fg_Ff(x)
                                if isinstance(r1, np.ndarray):
                                    if not np.allclose(r1, r2, atol=1e-6):
                                        results["composition_preservation"] = False
                                elif abs(r1 - r2) > 1e-6:
                                    results["composition_preservation"] = False
                            except Exception:
                                pass
                    except Exception:
                        pass

        results["is_functor"] = (results["identity_preservation"]
                                  and results["composition_preservation"])
        return results


class CategoricalDiscoveryEngine:
    """High-level engine for category-theoretic mathematical discovery.

    Combines morphism invariant finding, equivalence discovery,
    and functorial analysis to generate conjectures.
    """
    def __init__(self):
        self.conjecture_gen = ConjectureGenerator()
        self.invariant_finder = MorphismInvariantFinder()
        self.equiv_discovery = EquivalenceDiscovery()
        self.functor_discovery = FunctorialDiscovery()
        self.discoveries: List[Dict] = []

    def analyze_morphism(self, f: Callable, properties: Dict[str, Callable],
                          test_data: List[Any], name: str = "f") -> List[Conjecture]:
        """Analyze a morphism and generate conjectures about preserved properties."""
        preserved = self.invariant_finder.find_preserved_properties(
            f, properties, test_data, name)

        conjectures = []
        for p in preserved:
            c = Conjecture(
                f"Morphism '{name}' preserves property '{p['property']}' "
                f"(rate: {p['preservation_rate']:.2%})",
                ConjectureType.INVARIANCE,
                source="categorical_invariant_analysis",
            )
            c.add_evidence(f"Tested {p['n_tested']} elements")
            conjectures.append(c)

        self.conjecture_gen.conjectures.extend(conjectures)
        self.discoveries.append({
            "type": "morphism_analysis", "name": name,
            "preserved": [p["property"] for p in preserved],
        })
        return conjectures

    def analyze_equivalences(self, elements: List[Any],
                               equiv_fn: Callable,
                               name: str = "~") -> List[Conjecture]:
        """Discover equivalence classes and generate conjectures."""
        classes = self.equiv_discovery.find_equivalence_classes(elements, equiv_fn)

        conjectures = []
        if len(classes) < len(elements):
            c = Conjecture(
                f"Relation '{name}' partitions {len(elements)} elements into "
                f"{len(classes)} equivalence classes "
                f"(sizes: {sorted([len(c) for c in classes], reverse=True)[:5]})",
                ConjectureType.EQUIVALENCE,
                source="equivalence_discovery",
            )
            c.add_evidence(f"Found {len(classes)} distinct classes")
            conjectures.append(c)

        self.conjecture_gen.conjectures.extend(conjectures)
        return conjectures

    def discover_functorial_structure(self, obj_map: Callable,
                                        mor_map: Callable,
                                        objects: List[Any],
                                        morphisms: List[Tuple],
                                        test_data: List[Any],
                                        name: str = "F") -> List[Conjecture]:
        """Test if a mapping forms a functor and generate conjectures."""
        result = self.functor_discovery.test_functor_axioms(
            obj_map, mor_map, objects, morphisms, test_data)

        conjectures = []
        if result["is_functor"]:
            c = Conjecture(
                f"Mapping '{name}' satisfies functor axioms "
                f"(identity: {result['identity_preservation']}, "
                f"composition: {result['composition_preservation']})",
                ConjectureType.EQUIVALENCE,
                source="functorial_discovery",
            )
            c.add_evidence("Both functor axioms verified")
            c.confidence = 0.85
            conjectures.append(c)

        self.conjecture_gen.conjectures.extend(conjectures)
        return conjectures

    def report(self) -> str:
        return self.conjecture_gen.report()
