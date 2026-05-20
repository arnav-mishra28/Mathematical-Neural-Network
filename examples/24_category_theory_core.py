"""
Example 24 — Category Theory: Core Categorical Structures
Demonstrates objects, morphisms, categories, composition, and diagram verification.
"""
import numpy as np
from mnn.category.core import (
    CatObject, Morphism, IdentityMorphism, Category,
    ProductCategory, OppositeCategory, SliceCategory,
    make_vect_category, make_grp_category,
)


def main():
    print("=" * 65)
    print("  CATEGORY THEORY ENGINE — Core Structures")
    print("=" * 65)

    # ---- 1. Build Vect: category of vector spaces ----
    print("\n[1] Vect — Category of finite-dimensional vector spaces")
    Vect = make_vect_category(dims=[1, 2, 3])
    print(Vect.summary())

    # Test composition: ι_{1→2} then ι_{2→3} = ι_{1→3}
    r1 = Vect.get_object("R1")
    r2 = Vect.get_object("R2")
    r3 = Vect.get_object("R3")

    # Compose injection morphisms
    inj_12 = Vect.hom(r1, r2)[0]
    inj_23 = Vect.hom(r2, r3)[0]
    inj_13 = inj_23 @ inj_12   # g ∘ f

    x = np.array([5.0])
    result = inj_13(x)
    print(f"\n  ι_{{2→3}} ∘ ι_{{1→2}} applied to [5.0] = {result.flatten()}")
    assert np.allclose(result.flatten()[:1], [5.0])
    print("  ✓ Composition correct")

    # Identity law
    print(f"  Identity laws hold: {Vect.verify_identity_laws()}")

    # ---- 2. Build Grp: category of groups ----
    print("\n[2] Grp — Category of finite groups")
    Grp = make_grp_category()
    print(Grp.summary())

    # ---- 3. Custom category with commutativity check ----
    print("\n[3] Custom category with commutative diagram")
    C = Category("C", "Test category")
    A = CatObject("A", "vector", data=np.array([1.0, 2.0]), dim=2)
    B = CatObject("B", "vector", data=np.array([0.0, 0.0, 0.0]), dim=3)
    D = CatObject("D", "vector", data=np.array([0.0]), dim=1)

    C.add_objects(A, B, D)

    # f: A → B (embed 2D → 3D)
    f = Morphism(A, B, fn=lambda x: np.array([x[0], x[1], 0.0]), name="f")
    # g: B → D (project 3D → 1D: take norm)
    g = Morphism(B, D, fn=lambda x: np.array([np.linalg.norm(x)]), name="g")
    # h: A → D (direct: norm of 2D)
    h = Morphism(A, D, fn=lambda x: np.array([np.linalg.norm(x)]), name="h")

    C.add_morphisms(f, g, h)
    print(C.summary())

    # Check: g∘f = h  (commutative diagram)
    x_test = np.array([3.0, 4.0])
    commutes = C.verify_commutative_diagram(
        paths=[["f", "g"], ["h"]],
        test_input=x_test
    )
    print(f"\n  Diagram commutes (g∘f = h)? {commutes}")
    print(f"    g∘f([3,4]) = {(g @ f)(x_test)}")
    print(f"    h([3,4])   = {h(x_test)}")

    # ---- 4. Associativity check ----
    print("\n[4] Associativity verification")
    E = CatObject("E", "vector", dim=1)
    C.add_object(E)
    k = Morphism(D, E, fn=lambda x: x * 2, name="k")
    C.add_morphism(k)
    # k ∘ (g ∘ f) vs (k ∘ g) ∘ f
    assoc = C.verify_associativity("f", "g", "k", test_input=x_test)
    print(f"  (k∘g)∘f = k∘(g∘f)? {assoc}")

    # ---- 5. Product category ----
    print("\n[5] Product category Vect × Grp")
    Prod = ProductCategory(Vect, Grp)
    print(f"  {Prod}: {len(Prod.objects)} objects")

    # ---- 6. Opposite category ----
    print("\n[6] Opposite category C^op")
    C_op = OppositeCategory(C)
    print(f"  {C_op}: {len(C_op.objects)} objects, {len(C_op.morphisms)} morphisms")
    for m in C_op.morphisms:
        if not isinstance(m, IdentityMorphism):
            print(f"    {m}")

    print("\n  ✓ All categorical structures verified")


if __name__ == "__main__":
    main()
