"""mnn.category.functors — Functors and natural transformations.

A *functor* F: C → D maps a category C to a category D:
  - Objects:   A  ↦  F(A)
  - Morphisms: (f: A→B)  ↦  F(f): F(A)→F(B)
  - Preserves composition: F(g∘f) = F(g)∘F(f)
  - Preserves identity:    F(id_A) = id_{F(A)}

A *natural transformation* α: F ⇒ G assigns to each object A a morphism
α_A: F(A) → G(A) such that for every f: A → B the diagram commutes:
  G(f) ∘ α_A  =  α_B ∘ F(f)

MNN-specific functors bridge existing modules:
  GeometryToAlgebra   — manifold → metric tensor → algebra
  AlgebraToComputation — group → Cayley table → numeric
  DynamicsToLearning   — ODE → flow map learner
  UniversalBridge      — generic inter-module converter
"""
from __future__ import annotations
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from .core import CatObject, Morphism, IdentityMorphism, Category, CompositionError


# ---------------------------------------------------------------------------
# Functor
# ---------------------------------------------------------------------------

class Functor:
    """Covariant functor F: C → D.

    Parameters
    ----------
    source, target : Category
    obj_map  : callable  CatObject → CatObject
    mor_map  : callable  Morphism  → Morphism
    name     : str
    """
    def __init__(self, source: Category, target: Category,
                 obj_map: Callable[[CatObject], CatObject],
                 mor_map: Callable[[Morphism], Morphism],
                 name: str = "F"):
        self.source = source
        self.target = target
        self.obj_map = obj_map
        self.mor_map = mor_map
        self.name = name

    def apply_object(self, obj: CatObject) -> CatObject:
        mapped = self.obj_map(obj)
        self.target.add_object(mapped)
        return mapped

    def apply_morphism(self, m: Morphism) -> Morphism:
        mapped = self.mor_map(m)
        self.target.add_morphism(mapped)
        return mapped

    def apply_all(self):
        """Map every object and morphism in the source category."""
        for o in self.source.objects:
            self.apply_object(o)
        for m in self.source.morphisms:
            if not isinstance(m, IdentityMorphism):
                self.apply_morphism(m)

    # ---- Functor axiom checks ----
    def verify_identity(self) -> bool:
        """F(id_A) should be id_{F(A)} (structurally)."""
        for o in self.source.objects:
            id_A = IdentityMorphism(o)
            F_id = self.mor_map(id_A)
            F_A = self.obj_map(o)
            if F_id.domain != F_A or F_id.codomain != F_A:
                return False
        return True

    def verify_composition(self, f_name: str, g_name: str,
                           test_input: Any = None) -> bool:
        """Check F(g∘f) = F(g) ∘ F(f)."""
        f = self.source._morphisms[f_name]
        g = self.source._morphisms[g_name]
        gf = g.compose(f)
        F_gf = self.mor_map(gf)
        F_g = self.mor_map(g)
        F_f = self.mor_map(f)
        F_g_F_f = F_g.compose(F_f)
        if test_input is not None:
            r1 = F_gf(test_input)
            r2 = F_g_F_f(test_input)
            if isinstance(r1, np.ndarray):
                return bool(np.allclose(r1, r2, atol=1e-8))
            return r1 == r2
        return (F_gf.domain == F_g_F_f.domain and
                F_gf.codomain == F_g_F_f.codomain)

    def compose(self, other: "Functor") -> "Functor":
        """Functor composition: self ∘ other  (G ∘ F)."""
        return Functor(
            source=other.source, target=self.target,
            obj_map=lambda o, s=self, r=other: s.obj_map(r.obj_map(o)),
            mor_map=lambda m, s=self, r=other: s.mor_map(r.mor_map(m)),
            name=f"{self.name}∘{other.name}",
        )

    def __repr__(self):
        return f"Functor({self.name}: {self.source.name}→{self.target.name})"


class ContravariantFunctor(Functor):
    """Contravariant functor F: C → D — reverses morphism direction.

    F(f: A→B)  ↦  F(f): F(B)→F(A)
    """
    def apply_morphism(self, m: Morphism) -> Morphism:
        mapped = self.mor_map(m)
        # swap domain/codomain
        reversed_m = Morphism(
            domain=mapped.codomain, codomain=mapped.domain,
            fn=mapped.fn, name=f"{mapped.name}*",
            properties=mapped.properties,
        )
        self.target.add_morphism(reversed_m)
        return reversed_m


# ---------------------------------------------------------------------------
# Natural Transformation
# ---------------------------------------------------------------------------

class NaturalTransformation:
    """Natural transformation α: F ⇒ G between functors F, G: C → D.

    For each object A in C, provides a morphism α_A: F(A) → G(A)
    such that: G(f) ∘ α_A = α_B ∘ F(f)  for all f: A → B.

    Parameters
    ----------
    F, G : Functor
    components : dict mapping object name → Morphism (the component α_A)
    name : str
    """
    def __init__(self, F: Functor, G: Functor,
                 components: Dict[str, Morphism],
                 name: str = "α"):
        if F.source.name != G.source.name:
            raise ValueError("Functors must share source category")
        if F.target.name != G.target.name:
            raise ValueError("Functors must share target category")
        self.F = F
        self.G = G
        self.components = components
        self.name = name

    def component(self, obj: CatObject) -> Morphism:
        """α_A : F(A) → G(A)."""
        return self.components[obj.name]

    def verify_naturality(self, f: Morphism, test_input: Any = None) -> bool:
        """Check G(f) ∘ α_A = α_B ∘ F(f) for morphism f: A → B."""
        A, B = f.domain, f.codomain
        alpha_A = self.components.get(A.name)
        alpha_B = self.components.get(B.name)
        if alpha_A is None or alpha_B is None:
            return True  # skip if component not defined

        Ff = self.F.mor_map(f)
        Gf = self.G.mor_map(f)

        lhs = Gf.compose(alpha_A)   # G(f) ∘ α_A
        rhs = alpha_B.compose(Ff)    # α_B ∘ F(f)

        if test_input is not None:
            r1 = lhs(test_input)
            r2 = rhs(test_input)
            if isinstance(r1, np.ndarray):
                return bool(np.allclose(r1, r2, atol=1e-8))
            return r1 == r2
        return (lhs.domain == rhs.domain and lhs.codomain == rhs.codomain)

    def verify_all(self, test_input: Any = None) -> bool:
        """Verify naturality for all non-identity morphisms in source."""
        for m in self.F.source.morphisms:
            if isinstance(m, IdentityMorphism):
                continue
            if not self.verify_naturality(m, test_input):
                return False
        return True

    def horizontal_compose(self, other: "NaturalTransformation"
                           ) -> "NaturalTransformation":
        """Horizontal composition β ∗ α."""
        new_comps = {}
        for name, alpha_A in self.components.items():
            if name in other.components:
                beta_FA = other.components[name]
                new_comps[name] = beta_FA.compose(alpha_A)
        return NaturalTransformation(
            self.F, other.G, new_comps,
            name=f"{other.name}∗{self.name}",
        )

    def vertical_compose(self, other: "NaturalTransformation"
                         ) -> "NaturalTransformation":
        """Vertical composition β · α (β: G⇒H, α: F⇒G → β·α: F⇒H)."""
        new_comps = {}
        for name in self.components:
            if name in other.components:
                new_comps[name] = other.components[name].compose(self.components[name])
        return NaturalTransformation(
            self.F, other.G, new_comps,
            name=f"{other.name}·{self.name}",
        )

    def __repr__(self):
        return (f"NatTrans({self.name}: {self.F.name}⇒{self.G.name}, "
                f"components={len(self.components)})")


# ---------------------------------------------------------------------------
# MNN-specific functors: bridge between modules
# ---------------------------------------------------------------------------

class ForgetfulFunctor(Functor):
    """U: StructuredCategory → Set  — forgets structure, keeps underlying data.

    Example: Grp → Set (forget group operation, keep elements).
    """
    def __init__(self, source: Category, name: str = "U"):
        target = Category(f"Set_from_{source.name}", "Underlying sets")
        super().__init__(
            source=source, target=target,
            obj_map=lambda o: CatObject(f"|{o.name}|", "set",
                                         data=o.data, dim=o.dim),
            mor_map=lambda m: Morphism(
                CatObject(f"|{m.domain.name}|", "set", data=m.domain.data),
                CatObject(f"|{m.codomain.name}|", "set", data=m.codomain.data),
                fn=m.fn, name=f"U({m.name})",
            ),
            name=name,
        )


class FreeObjectFunctor(Functor):
    """F: Set → StructuredCategory — free construction (left adjoint to forgetful).

    Example: Set → Vect (free vector space on a set).
    """
    def __init__(self, target: Category, constructor: Callable,
                 name: str = "Free"):
        source = Category("Set", "Sets")
        super().__init__(
            source=source, target=target,
            obj_map=lambda o: constructor(o),
            mor_map=lambda m: Morphism(
                constructor(m.domain), constructor(m.codomain),
                fn=m.fn, name=f"Free({m.name})",
            ),
            name=name,
        )


class GeometryToAlgebraFunctor(Functor):
    """Geometry → Algebra: manifold ↦ (metric tensor, Christoffel symbols).

    Maps Riemannian manifolds to their algebraic invariants.
    """
    def __init__(self, geom_cat: Category, alg_cat: Category):
        def obj_map(obj: CatObject) -> CatObject:
            if obj.kind == "manifold" and obj.data is not None:
                m = obj.data
                try:
                    metric = m.metric_tensor()
                    return CatObject(
                        f"Alg({obj.name})", "tensor",
                        data={"metric": metric, "manifold": m},
                        dim=obj.dim,
                        metadata={"source_manifold": obj.name},
                    )
                except Exception:
                    pass
            return CatObject(f"Alg({obj.name})", "generic", data=obj.data, dim=obj.dim)

        def mor_map(m: Morphism) -> Morphism:
            return Morphism(
                obj_map(m.domain), obj_map(m.codomain),
                fn=m.fn, name=f"Alg({m.name})",
            )

        super().__init__(geom_cat, alg_cat, obj_map, mor_map, name="GeomToAlg")


class AlgebraToComputationFunctor(Functor):
    """Algebra → Computation: group ↦ Cayley table (numeric matrix).

    Maps algebraic structures to their computational representations.
    """
    def __init__(self, alg_cat: Category, comp_cat: Category):
        def obj_map(obj: CatObject) -> CatObject:
            if obj.kind == "group" and obj.data is not None:
                g = obj.data
                try:
                    table = g.cayley_table()
                    return CatObject(
                        f"Comp({obj.name})", "matrix",
                        data=table, dim=g.order,
                        metadata={"source_group": obj.name, "abelian": g.is_abelian()},
                    )
                except Exception:
                    pass
            return CatObject(f"Comp({obj.name})", "generic", data=obj.data, dim=obj.dim)

        def mor_map(m: Morphism) -> Morphism:
            return Morphism(
                obj_map(m.domain), obj_map(m.codomain),
                fn=m.fn, name=f"Comp({m.name})",
            )

        super().__init__(alg_cat, comp_cat, obj_map, mor_map, name="AlgToComp")


class DynamicsToLearningFunctor(Functor):
    """Dynamics → Learning: ODE system ↦ flow-map neural learner.

    Maps dynamical systems to their neural-network learned representations.
    """
    def __init__(self, dyn_cat: Category, learn_cat: Category):
        def obj_map(obj: CatObject) -> CatObject:
            if obj.kind == "dynamical_system" and obj.data is not None:
                return CatObject(
                    f"Learn({obj.name})", "neural_network",
                    data={"ode": obj.data}, dim=obj.dim,
                    metadata={"source_system": obj.name},
                )
            return CatObject(f"Learn({obj.name})", "generic", data=obj.data, dim=obj.dim)

        def mor_map(m: Morphism) -> Morphism:
            return Morphism(
                obj_map(m.domain), obj_map(m.codomain),
                fn=m.fn, name=f"Learn({m.name})",
            )

        super().__init__(dyn_cat, learn_cat, obj_map, mor_map, name="DynToLearn")


class UniversalBridgeFunctor(Functor):
    """Generic bridge functor between any two MNN categories.

    Uses a user-supplied converter dict: {source_kind: (target_kind, convert_fn)}.
    """
    def __init__(self, source: Category, target: Category,
                 converters: Dict[str, Tuple[str, Callable]],
                 name: str = "Bridge"):
        self._converters = converters

        def obj_map(obj: CatObject) -> CatObject:
            if obj.kind in converters:
                tkind, cfn = converters[obj.kind]
                return CatObject(
                    f"{name}({obj.name})", tkind,
                    data=cfn(obj.data) if obj.data is not None else None,
                    dim=obj.dim,
                )
            return CatObject(f"{name}({obj.name})", obj.kind, data=obj.data, dim=obj.dim)

        def mor_map(m: Morphism) -> Morphism:
            return Morphism(
                obj_map(m.domain), obj_map(m.codomain),
                fn=m.fn, name=f"{name}({m.name})",
            )

        super().__init__(source, target, obj_map, mor_map, name=name)
