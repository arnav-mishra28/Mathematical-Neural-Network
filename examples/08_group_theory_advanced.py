"""
Example 08 — Advanced Group Theory
====================================
Derived series, Sylow theorems, composition series,
homomorphisms, semi-direct products, representation theory.
"""
import sys; sys.path.insert(0, "..")
import numpy as np
from mnn.algebra.groups import Group
from mnn.advanced.group_theory import (
    FiniteGroupAnalyzer, GroupHomomorphism, GroupExtension, RepresentationTheory
)

print("=" * 56)
print("  MNN Example 08 — Advanced Group Theory")
print("=" * 56)

# ── 1. Structural analysis: Z_12 ─────────────────────────────
print("\n[1] Structural analysis of Z₁₂")
Z12  = Group.cyclic(12)
ana  = FiniteGroupAnalyzer(Z12)
ds   = ana.derived_series()
print(f"    Derived series lengths: {[len(s) for s in ds]}")
print(f"    Solvable: {ana.is_solvable()}")
print(f"    Nilpotent: {ana.is_nilpotent()}")
print(f"    Nilpotency class: {ana.nilpotency_class()}")
print(ana.structural_report())

# ── 2. S₄ analysis ───────────────────────────────────────────
print("\n[2] Structural analysis of S₄")
S4   = Group.symmetric(4)
ana4 = FiniteGroupAnalyzer(S4)
print(f"    Solvable: {ana4.is_solvable()}")
print(f"    Derived series: {[len(s) for s in ana4.derived_series()]}")
print(f"    Composition factors: {ana4.composition_factors()}")

# ── 3. Sylow theorems: S₃ ────────────────────────────────────
print("\n[3] Sylow theorems for S₃")
S3   = Group.symmetric(3)
ana3 = FiniteGroupAnalyzer(S3)
for p in [2, 3]:
    v = ana3.sylow_verification(p)
    print(f"    p={p}: Sylow order={v['sylow_order']}, "
          f"n_p={v['n_p']}, thm1={v['thm1_exists']}, "
          f"thm3_mod={v['thm3_mod_p']}, thm3_div={v['thm3_divides']}")

# ── 4. Group homomorphism ─────────────────────────────────────
print("\n[4] Homomorphism Z₄ → Z₂ (reduction mod 2)")
Z4   = Group.cyclic(4)
Z2   = Group.cyclic(2)
phi  = lambda x: x % 2
hom  = GroupHomomorphism(Z4, Z2, phi, "φ")
print(f"    Is homomorphism : {hom.is_homomorphism()}")
print(f"    Kernel          : {hom.kernel()}")
print(f"    Image           : {hom.image()}")
print(f"    Injective       : {hom.is_injective()}")
print(f"    Surjective      : {hom.is_surjective()}")
print(f"    {hom.first_isomorphism_theorem()}")

# ── 5. Semi-direct product: D₃ ≅ Z₃ ⋊ Z₂ ────────────────────
print("\n[5] Semi-direct product Z₃ ⋊ Z₂ ≅ D₃")
Z3   = Group.cyclic(3)
Z2b  = Group.cyclic(2)
# Action: flip sends n → -n mod 3
action = lambda q, n: (-n) % 3 if q == 1 else n
D3_sdp = GroupExtension.semidirect_product(Z3, Z2b, action)
print(f"    {D3_sdp}")
print(f"    Order : {D3_sdp.order}  (expected 6)")
print(f"    Abelian: {D3_sdp.is_abelian()}  (expected False)")

# ── 6. Representation theory ──────────────────────────────────
print("\n[6] Representation theory of Z₆")
Z6   = Group.cyclic(6)
RT   = RepresentationTheory(Z6)
triv = RT.trivial_representation()
reg  = RT.regular_representation()
print(f"    Trivial rep is rep : {triv.is_representation()}")
print(f"    Regular rep degree : {reg.degree}")
print(f"    # irreducibles     : {RT.number_of_irreducibles()}  (= # conj. classes)")
print(f"    Dimension formula  : {RT.dimension_formula()}")

# Character table
tab, elems = RT.character_table_abelian()
print(f"    Character table shape: {tab.shape}")
print(f"    Orthogonality check : {RT.verify_orthogonality()}")

# Irreducibility of trivial rep
chi_t = triv.character()
norm  = RT.character_inner_product(chi_t, chi_t)
print(f"    ⟨χ_triv, χ_triv⟩ = {norm.real:.4f}  (expected 1 = irreducible)")

print("\n[OK] Example 08 complete.")
