"""mnn.category.core — Core categorical structures.

Implements the foundational abstractions of category theory:
  - Objects: typed mathematical entities (vectors, tensors, manifolds, groups, fields)
  - Morphisms: structure-preserving maps between objects (f: A → B)
  - Categories: collections of objects and morphisms with composition and identity
  - Products/Coproducts: categorical constructions for combining objects
  - Slice categories, opposite categories, and diagram commutativity checking

This module does NOT use neural networks — it is pure categorical algebra.
The neural extension lives in mnn.category.neural.
"""
from __future__ import annotations
import numpy as np
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union, TypeVar, Generic,
)
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib


class CompositionError(Exception):
    """Raised when morphism composition is invalid (codomain ≠ domain)."""


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------

class CatObject:
    """An object in a category.

    Objects are typed mathematical entities. The 'kind' tag lets the
    categorical engine know what module the object belongs to.

    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. "R³", "S²", "Z_4").
    kind : str
        Semantic type tag.  Recognised kinds:
        'vector', 'scalar_field', 'vector_field', 'tensor',
        'manifold', 'group', 'lie_group', 'topological_space',
        'neural_network', 'pde_solution', 'dynamical_system', 'generic'.
    data : Any, optional
        The wrapped MNN object (ScalarField, Group, MNNNetwork, …).
    dim : int, optional
        Dimensionality hint (used by functors for shape inference).
    metadata : dict, optional
        Arbitrary extra info (coordinate names, parameters, …).
    """
    __slots__ = ("name", "kind", "data", "dim", "metadata", "_uid")

    def __init__(self, name: str, kind: str = "generic", data: Any = None,
                 dim: Optional[int] = None, metadata: Optional[Dict] = None):
        self.name = name
        self.kind = kind
        self.data = data
        self.dim = dim
        self.metadata = metadata or {}
        self._uid = hashlib.md5(f"{name}:{kind}:{id(data)}".encode()).hexdigest()[:12]

    # Two objects are 'the same' if they wrap the same data, or share uid
    def __eq__(self, other):
        if not isinstance(other, CatObject):
            return NotImplemented
        return self._uid == other._uid

    def __hash__(self):
        return hash(self._uid)

    def __repr__(self):
        dim_s = f", dim={self.dim}" if self.dim is not None else ""
        return f"Obj({self.name}: {self.kind}{dim_s})"


# ---------------------------------------------------------------------------
# Morphisms
# ---------------------------------------------------------------------------

class Morphism:
    """A morphism f: A → B in a category.

    Wraps a callable *and* tracks domain/codomain so the categorical
    engine can verify composition and commutative diagrams.

    Parameters
    ----------
    domain : CatObject   — source object A
    codomain : CatObject  — target object B
    fn : callable         — the actual map  (receives A.data, returns B-compatible data)
    name : str            — human-readable label
    properties : set      — optional tags like {'injective', 'surjective', 'iso', 'endo'}
    """
    __slots__ = ("domain", "codomain", "fn", "name", "properties")

    def __init__(self, domain: CatObject, codomain: CatObject,
                 fn: Callable, name: str = "f",
                 properties: Optional[Set[str]] = None):
        self.domain = domain
        self.codomain = codomain
        self.fn = fn
        self.name = name
        self.properties = properties or set()

    def __call__(self, x: Any = None) -> Any:
        """Apply the morphism.  If *x* is None, uses domain.data."""
        return self.fn(x if x is not None else self.domain.data)

    def compose(self, other: "Morphism") -> "Morphism":
        """self ∘ other  (apply *other* first, then *self*).

        Requires other.codomain == self.domain.
        """
        if other.codomain != self.domain:
            raise CompositionError(
                f"Cannot compose {self.name} ∘ {other.name}: "
                f"codomain({other.name})={other.codomain} ≠ domain({self.name})={self.domain}"
            )
        composed_fn = lambda x, _s=self, _o=other: _s.fn(_o.fn(x))
        return Morphism(
            domain=other.domain, codomain=self.codomain,
            fn=composed_fn, name=f"{self.name}∘{other.name}",
            properties=self.properties & other.properties,  # intersection
        )

    def __matmul__(self, other: "Morphism") -> "Morphism":
        """Operator shorthand: g @ f  ≡  g.compose(f)."""
        return self.compose(other)

    def is_endomorphism(self) -> bool:
        return self.domain == self.codomain

    def is_isomorphism(self) -> bool:
        return "iso" in self.properties

    def __repr__(self):
        props = f" [{','.join(sorted(self.properties))}]" if self.properties else ""
        return f"{self.name}: {self.domain.name}→{self.codomain.name}{props}"


class IdentityMorphism(Morphism):
    """id_A : A → A.  The identity morphism on an object."""
    def __init__(self, obj: CatObject):
        super().__init__(
            domain=obj, codomain=obj,
            fn=lambda x: x,
            name=f"id_{obj.name}",
            properties={"iso", "endo"},
        )


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

class Category:
    """A category C = (Ob(C), Hom(C), ∘, id).

    Parameters
    ----------
    name : str           — name of the category (e.g. "Vect", "Grp", "Man")
    description : str    — free-form description
    """
    def __init__(self, name: str = "C", description: str = ""):
        self.name = name
        self.description = description
        self._objects: Dict[str, CatObject] = {}
        self._morphisms: Dict[str, Morphism] = {}
        self._hom_sets: Dict[Tuple[str, str], List[Morphism]] = defaultdict(list)

    # ---- Object management ----
    def add_object(self, obj: CatObject) -> CatObject:
        self._objects[obj.name] = obj
        # auto-register the identity morphism
        self._register_morphism(IdentityMorphism(obj))
        return obj

    def add_objects(self, *objs: CatObject):
        for o in objs:
            self.add_object(o)

    def get_object(self, name: str) -> CatObject:
        return self._objects[name]

    @property
    def objects(self) -> List[CatObject]:
        return list(self._objects.values())

    # ---- Morphism management ----
    def _register_morphism(self, m: Morphism):
        key = m.name
        self._morphisms[key] = m
        self._hom_sets[(m.domain.name, m.codomain.name)].append(m)

    def add_morphism(self, m: Morphism) -> Morphism:
        if m.domain.name not in self._objects:
            self.add_object(m.domain)
        if m.codomain.name not in self._objects:
            self.add_object(m.codomain)
        self._register_morphism(m)
        return m

    def add_morphisms(self, *ms: Morphism):
        for m in ms:
            self.add_morphism(m)

    def hom(self, A: Union[str, CatObject], B: Union[str, CatObject]) -> List[Morphism]:
        """Hom(A,B) — all registered morphisms from A to B."""
        a = A.name if isinstance(A, CatObject) else A
        b = B.name if isinstance(B, CatObject) else B
        return list(self._hom_sets[(a, b)])

    @property
    def morphisms(self) -> List[Morphism]:
        return list(self._morphisms.values())

    # ---- Composition ----
    def compose(self, g_name: str, f_name: str, result_name: Optional[str] = None
                ) -> Morphism:
        """Register g ∘ f in the category."""
        g = self._morphisms[g_name]
        f = self._morphisms[f_name]
        h = g.compose(f)
        if result_name:
            h.name = result_name
        self._register_morphism(h)
        return h

    # ---- Verification ----
    def verify_identity_laws(self) -> bool:
        """Check id_B ∘ f = f = f ∘ id_A for all f: A → B."""
        for m in self._morphisms.values():
            if isinstance(m, IdentityMorphism):
                continue
            id_a = IdentityMorphism(m.domain)
            id_b = IdentityMorphism(m.codomain)
            # We can only verify structurally (name + domain/codomain)
            h1 = id_b.compose(m)
            h2 = m.compose(id_a)
            if h1.domain != m.domain or h1.codomain != m.codomain:
                return False
            if h2.domain != m.domain or h2.codomain != m.codomain:
                return False
        return True

    def verify_associativity(self, f_name: str, g_name: str, h_name: str,
                              test_input: Any = None) -> bool:
        """Check h∘(g∘f) = (h∘g)∘f numerically."""
        f = self._morphisms[f_name]
        g = self._morphisms[g_name]
        h = self._morphisms[h_name]
        gf = g.compose(f)
        hg = h.compose(g)
        lhs = h.compose(gf)
        rhs = hg.compose(f)
        if test_input is None:
            return lhs.domain == rhs.domain and lhs.codomain == rhs.codomain
        r1 = lhs(test_input)
        r2 = rhs(test_input)
        if isinstance(r1, np.ndarray):
            return np.allclose(r1, r2, atol=1e-8)
        return r1 == r2

    def verify_commutative_diagram(self, paths: List[List[str]],
                                    test_input: Any = None) -> bool:
        """Check that multiple composition paths give the same result.

        *paths*: e.g. [["f","g"], ["h"]] checks g∘f = h.
        """
        def compose_path(names: List[str]) -> Morphism:
            result = self._morphisms[names[0]]
            for n in names[1:]:
                result = self._morphisms[n].compose(result)
            return result

        composed = [compose_path(p) for p in paths]
        # structural check
        if not all(c.domain == composed[0].domain and c.codomain == composed[0].codomain
                   for c in composed):
            return False
        # numeric check if input provided
        if test_input is not None:
            results = [c(test_input) for c in composed]
            ref = results[0]
            for r in results[1:]:
                if isinstance(ref, np.ndarray):
                    if not np.allclose(ref, r, atol=1e-8):
                        return False
                elif ref != r:
                    return False
        return True

    # ---- Special constructions ----
    def product_object(self, A: CatObject, B: CatObject) -> CatObject:
        """A × B as a categorical product."""
        prod = CatObject(
            name=f"{A.name}×{B.name}", kind="product",
            data=(A.data, B.data),
            dim=(A.dim or 0) + (B.dim or 0),
            metadata={"factors": (A.name, B.name)},
        )
        self.add_object(prod)
        # Projection morphisms
        self.add_morphism(Morphism(prod, A, lambda d: d[0], name=f"π₁_{A.name}"))
        self.add_morphism(Morphism(prod, B, lambda d: d[1], name=f"π₂_{B.name}"))
        return prod

    def coproduct_object(self, A: CatObject, B: CatObject) -> CatObject:
        """A ⊔ B as a categorical coproduct."""
        coprod = CatObject(
            name=f"{A.name}⊔{B.name}", kind="coproduct",
            data=(A.data, B.data),
            metadata={"summands": (A.name, B.name)},
        )
        self.add_object(coprod)
        self.add_morphism(Morphism(A, coprod, lambda d: ("left", d), name=f"ι₁_{A.name}"))
        self.add_morphism(Morphism(B, coprod, lambda d: ("right", d), name=f"ι₂_{B.name}"))
        return coprod

    # ---- Reporting ----
    def summary(self) -> str:
        n_obj = len(self._objects)
        n_mor = len(self._morphisms)
        n_id = sum(1 for m in self._morphisms.values() if isinstance(m, IdentityMorphism))
        lines = [
            f"Category: {self.name}",
            f"  Objects ({n_obj}): {', '.join(self._objects.keys())}",
            f"  Morphisms ({n_mor}, {n_id} identities):",
        ]
        for m in self._morphisms.values():
            if not isinstance(m, IdentityMorphism):
                lines.append(f"    {m}")
        return "\n".join(lines)

    def __repr__(self):
        return f"Category({self.name}, |Ob|={len(self._objects)}, |Mor|={len(self._morphisms)})"


# ---------------------------------------------------------------------------
# Derived categorical constructions
# ---------------------------------------------------------------------------

class ProductCategory(Category):
    """Product category C × D."""
    def __init__(self, C: Category, D: Category):
        super().__init__(name=f"{C.name}×{D.name}",
                         description=f"Product of {C.name} and {D.name}")
        self.C = C
        self.D = D
        for a in C.objects:
            for b in D.objects:
                self.add_object(CatObject(f"({a.name},{b.name})", "product_pair",
                                           data=(a.data, b.data)))

    def __repr__(self):
        return f"ProductCategory({self.C.name}×{self.D.name})"


class OppositeCategory(Category):
    """Opposite category C^op — reverses all morphisms."""
    def __init__(self, C: Category):
        super().__init__(name=f"{C.name}ᵒᵖ",
                         description=f"Opposite of {C.name}")
        self.original = C
        for o in C.objects:
            self.add_object(o)
        for m in C.morphisms:
            if not isinstance(m, IdentityMorphism):
                self.add_morphism(Morphism(
                    domain=m.codomain, codomain=m.domain,
                    fn=m.fn,  # the 'reversal' is structural, not computational
                    name=f"{m.name}ᵒᵖ",
                ))


class SliceCategory(Category):
    """Slice category C/A — objects are morphisms f: X → A in C."""
    def __init__(self, C: Category, A: CatObject):
        super().__init__(name=f"{C.name}/{A.name}",
                         description=f"Slice over {A.name}")
        self.base = C
        self.apex = A
        for m in C.morphisms:
            if m.codomain == A and not isinstance(m, IdentityMorphism):
                obj = CatObject(f"({m.domain.name}→{A.name})", "slice_object",
                                 data=m, metadata={"base_morphism": m.name})
                self.add_object(obj)


# ---------------------------------------------------------------------------
# Pre-built MNN categories (factories)
# ---------------------------------------------------------------------------

def make_vect_category(dims: Optional[List[int]] = None) -> Category:
    """Vect — category of finite-dimensional real vector spaces & linear maps."""
    C = Category("Vect", "Finite-dimensional real vector spaces")
    dims = dims or [1, 2, 3]
    for d in dims:
        C.add_object(CatObject(f"R{d}", "vector", dim=d, data=np.zeros(d)))
    # canonical inclusions / projections where dimensions nest
    for i, d1 in enumerate(dims):
        for d2 in dims[i+1:]:
            inj = np.zeros((d2, d1))
            inj[:d1, :d1] = np.eye(d1)
            C.add_morphism(Morphism(
                C.get_object(f"R{d1}"), C.get_object(f"R{d2}"),
                fn=lambda x, M=inj: M @ np.asarray(x),
                name=f"ι_{d1}→{d2}", properties={"injective"},
            ))
            proj = np.zeros((d1, d2))
            proj[:d1, :d1] = np.eye(d1)
            C.add_morphism(Morphism(
                C.get_object(f"R{d2}"), C.get_object(f"R{d1}"),
                fn=lambda x, M=proj: M @ np.asarray(x),
                name=f"π_{d2}→{d1}", properties={"surjective"},
            ))
    return C


def make_grp_category() -> Category:
    """Grp — small category of some standard finite groups."""
    from mnn.algebra.groups import Group
    C = Category("Grp", "Category of finite groups and group homomorphisms")
    for n in [2, 3, 4, 6]:
        g = Group.cyclic(n)
        C.add_object(CatObject(f"Z_{n}", "group", data=g, dim=n))
    return C


def make_man_category() -> Category:
    """Man — category of Riemannian manifolds."""
    C = Category("Man", "Category of Riemannian manifolds and smooth maps")
    return C


def make_top_category() -> Category:
    """Top — category of topological spaces."""
    C = Category("Top", "Category of topological spaces and continuous maps")
    return C
