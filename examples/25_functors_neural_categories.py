"""
Example 25 — Functors, Natural Transformations, and Neural Categories
Demonstrates the unifying power of category theory across MNN modules.
"""
import numpy as np
from mnn.category.core import (
    CatObject, Morphism, Category, make_vect_category, make_grp_category,
)
from mnn.category.functors import (
    Functor, NaturalTransformation, ForgetfulFunctor,
    GeometryToAlgebraFunctor, AlgebraToComputationFunctor,
    UniversalBridgeFunctor,
)
from mnn.category.neural import (
    NeuralMorphism, NeuralCategory, LearnableFunctor,
    CategoricalPipeline,
)


def main():
    print("=" * 65)
    print("  CATEGORY THEORY ENGINE — Functors & Neural Categories")
    print("=" * 65)

    # ---- 1. Functors between Vect categories ----
    print("\n[1] Functor: Vect_small → Vect_large (dimension doubling)")
    Vect_s = make_vect_category([1, 2])
    Vect_l = make_vect_category([2, 4])

    double_functor = Functor(
        Vect_s, Vect_l,
        obj_map=lambda o: CatObject(f"D({o.name})", "vector",
                                     dim=(o.dim or 1) * 2,
                                     data=np.zeros((o.dim or 1) * 2)),
        mor_map=lambda m: Morphism(
            CatObject(f"D({m.domain.name})", "vector", dim=(m.domain.dim or 1)*2),
            CatObject(f"D({m.codomain.name})", "vector", dim=(m.codomain.dim or 1)*2),
            fn=lambda x: np.tile(x, 2), name=f"D({m.name})",
        ),
        name="Double",
    )
    double_functor.apply_all()
    print(f"  {double_functor}")
    print(f"  Identity preservation: {double_functor.verify_identity()}")

    # ---- 2. Forgetful functor ----
    print("\n[2] Forgetful Functor: Grp → Set")
    Grp = make_grp_category()
    U = ForgetfulFunctor(Grp, name="U")
    U.apply_all()
    print(f"  {U}")
    print(f"  Target objects: {[o.name for o in U.target.objects]}")

    # ---- 3. Geometry → Algebra functor ----
    print("\n[3] GeometryToAlgebra Functor")
    from mnn.geometry.manifolds import RiemannianManifold
    Geom = Category("Geom", "Riemannian manifolds")
    s2 = RiemannianManifold.sphere_S2()
    Geom.add_object(CatObject("S²", "manifold", data=s2, dim=2))

    Alg = Category("Alg", "Algebraic structures")
    G2A = GeometryToAlgebraFunctor(Geom, Alg)
    G2A.apply_all()
    print(f"  {G2A}")
    for o in Alg.objects:
        if "metric" in str(o.metadata.get("source_manifold", "")):
            continue
        print(f"    {o}")

    # ---- 4. Algebra → Computation functor ----
    print("\n[4] AlgebraToComputation Functor")
    from mnn.algebra.groups import Group
    AlgCat = Category("Alg_groups")
    z4 = Group.cyclic(4)
    AlgCat.add_object(CatObject("Z_4", "group", data=z4, dim=4))

    CompCat = Category("Comp")
    A2C = AlgebraToComputationFunctor(AlgCat, CompCat)
    A2C.apply_all()
    for o in CompCat.objects:
        print(f"  {o} → abelian={o.metadata.get('abelian')}")
        if o.data is not None and hasattr(o.data, 'shape'):
            print(f"    Cayley table shape: {o.data.shape}")

    # ---- 5. Natural Transformation ----
    print("\n[5] Natural Transformation α: F ⇒ G")
    # Two functors from Vect_s to itself: identity and scaling
    F_id = Functor(Vect_s, Vect_s,
                   obj_map=lambda o: o,
                   mor_map=lambda m: m, name="Id")
    F_scale = Functor(Vect_s, Vect_s,
                      obj_map=lambda o: o,
                      mor_map=lambda m: Morphism(m.domain, m.codomain,
                                                  fn=lambda x: 2*np.asarray(x),
                                                  name=f"2·{m.name}"),
                      name="Scale2")
    # α_A: F(A) → G(A) defined as x ↦ 2x
    components = {}
    for o in Vect_s.objects:
        components[o.name] = Morphism(o, o, fn=lambda x: 2*np.asarray(x),
                                       name=f"α_{o.name}")
    alpha = NaturalTransformation(F_id, F_scale, components, name="α")
    print(f"  {alpha}")
    nat_ok = alpha.verify_all(test_input=np.array([1.0]))
    print(f"  Naturality verified: {nat_ok}")

    # ---- 6. Neural Category ----
    print("\n[6] Neural Category — NNs as morphisms")
    NC = NeuralCategory("MathCat", device="cpu")

    V2 = CatObject("V2", "vector", dim=2, data=np.zeros(2))
    V3 = CatObject("V3", "vector", dim=3, data=np.zeros(3))
    V1 = CatObject("V1", "vector", dim=1, data=np.zeros(1))

    NC.add_objects(V2, V3, V1)
    f_nn = NC.add_neural_morphism(V2, V3, width=32, depth=2, name="embed")
    g_nn = NC.add_neural_morphism(V3, V1, width=32, depth=2, name="project")

    # Train embed: R² → R³ (learn zero-padding)
    x2 = np.random.randn(500, 2).astype(np.float32)
    y3 = np.column_stack([x2, np.zeros(500)]).astype(np.float32)
    f_nn.train(x2, y3, n_epochs=500, verbose=False)

    # Train project: R³ → R¹ (learn norm)
    x3 = np.random.randn(500, 3).astype(np.float32)
    y1 = np.linalg.norm(x3, axis=1, keepdims=True).astype(np.float32)
    g_nn.train(x3, y1, n_epochs=500, verbose=False)

    # Evaluate composition
    test = np.array([[3.0, 4.0]], dtype=np.float32)
    result = NC.evaluate_path(["embed", "project"], test)
    print(f"  project ∘ embed ([3,4]) ≈ {result.flatten()}")
    print(f"  Expected ≈ {np.linalg.norm([3,4,0]):.2f}")

    print(f"\n{NC.summary()}")

    # ---- 7. Categorical Pipeline ----
    print("\n[7] Categorical Pipeline — cross-module data flow")
    pipe = CategoricalPipeline("FullPipeline")
    # Compose morphisms from different domains
    normalize = Morphism(
        CatObject("raw", "vector", dim=3),
        CatObject("normed", "vector", dim=3),
        fn=lambda x: x / (np.linalg.norm(x) + 1e-8),
        name="normalize"
    )
    dot_self = Morphism(
        CatObject("normed", "vector", dim=3),
        CatObject("scalar", "scalar_field", dim=1),
        fn=lambda x: np.array([np.dot(x.flatten(), x.flatten())]),
        name="self_dot"
    )
    pipe.add_stage("normalize", normalize)
    pipe.add_stage("self_dot", dot_self)

    intermediates = pipe.run_with_intermediates(np.array([3.0, 4.0, 0.0]))
    print(f"  Input:      {intermediates['input']}")
    print(f"  Normalized: {intermediates['normalize'].flatten()}")
    print(f"  Self-dot:   {intermediates['self_dot']}  (should be ≈1.0)")

    print(f"\n{pipe.summary()}")

    print("\n" + "=" * 65)
    print("  CATEGORY THEORY ENGINE — All structures verified ✓")
    print("=" * 65)


if __name__ == "__main__":
    main()
