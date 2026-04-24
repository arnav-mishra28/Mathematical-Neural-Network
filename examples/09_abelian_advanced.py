"""
Example 09 — Advanced Abelian Theory
=======================================
Elliptic curves, Jacobians, Abelian varieties, theta divisors.
"""
import sys; sys.path.insert(0, "..")
import numpy as np
from mnn.advanced.abelian import (
    EllipticCurve, JacobiVariety, AbelianVariety, ThetaDivisor
)

print("=" * 56)
print("  MNN Example 09 — Advanced Abelian Theory")
print("=" * 56)

# ── 1. Elliptic curve basics ──────────────────────────────────
print("\n[1] Elliptic curve E: y² = x³ − x")
E = EllipticCurve(a=-1., b=0.)
print(f"    {E}")
print(f"    Discriminant : {E.discriminant():.4f}")
print(f"    j-invariant  : {E.j_invariant():.4f}")

pts = E.sample_points((-2, 2), n=100)
print(f"    Sample points: {len(pts)} on curve")

# Verify a point
if len(pts) > 0:
    P_arr = pts[0]
    P = __import__('mnn.advanced.abelian.elliptic', fromlist=['Point']).Point(P_arr[0], P_arr[1])
    print(f"    Point P = {P}")
    print(f"    On curve: {E.is_on_curve(P)}")

# ── 2. Group law ──────────────────────────────────────────────
print("\n[2] Elliptic curve group law")
from mnn.advanced.abelian.elliptic import Point
# E: y² = x³ - x, take P = (0, 0) which is a 2-torsion point
P0 = Point(0., 0.)
print(f"    P = {P0}")
neg_P = E.negate(P0)
print(f"    -P = {neg_P}")
double_P = E.add(P0, P0)
print(f"    2P = {double_P}  (should be O since y=0 → 2-torsion)")

# Non-torsion addition
P1 = Point(-1., 0.)  # also 2-torsion
Q  = Point(2., np.sqrt(6.))  # approximate point on y² = x³ - x → 8-0 = 6 ✓
print(f"    Q = {Q}, on curve: {E.is_on_curve(Q, tol=1e-3)}")

# ── 3. Scalar multiplication ──────────────────────────────────
print("\n[3] Scalar multiplication on E: y² = x³ + x + 1")
E2  = EllipticCurve(a=1., b=1.)
pts2 = E2.sample_points((-1, 3), n=500)
if len(pts2) > 10:
    P   = Point(pts2[50][0], pts2[50][1])
    P2  = E2.scalar_mult(2, P)
    P3  = E2.scalar_mult(3, P)
    print(f"    P  = {P}")
    print(f"    2P = {P2}")
    print(f"    3P = {P3}")
    check = E2.add(P2, P)
    print(f"    2P + P == 3P: {abs(check.x - P3.x) < 1e-6 if not check.is_infinity else False}")

# ── 4. Over finite field F_7 ──────────────────────────────────
print("\n[4] E(F₇): y² = x³ − x (mod 7)")
Fp_pts = E.points_over_Fp(7)
print(f"    #E(F₇) = {len(Fp_pts)} (Hasse bound: |#E - 8| ≤ 2√7 ≈ 5.3)")
print(f"    Points: {Fp_pts[:6]}...")

# ── 5. g₂, g₃ invariants ─────────────────────────────────────
print("\n[5] Weierstrass invariants g₂, g₃")
g2, g3 = E.g2_g3_invariants()
print(f"    g₂ = {g2:.4f},  g₃ = {g3:.4f}")

# ── 6. Jacobi variety (genus 2) ───────────────────────────────
print("\n[6] Jacobi variety of a genus-2 curve")
Omega = np.array([[1.1j, 0.3j], [0.3j, 1.4j]])
J2    = JacobiVariety(genus=2, period_matrix=Omega)
print(f"    {J2}")
z_test = np.array([0.1+0.2j, 0.3+0.1j])
th     = J2.theta_function(z_test, N=3)
print(f"    Θ(z|Ω) = {th:.6f}")
print(f"    Principally polarized: {J2.is_principally_polarized()}")

# ── 7. Abelian variety ────────────────────────────────────────
print("\n[7] Abelian variety (g=2)")
A = AbelianVariety(Omega, name="A₂")
print(f"    {A}")
print(f"    Siegel half-space: {A.is_in_siegel_half_space()}")
print(f"    Rosati self-dual: {A.rosati_involution(np.eye(2)).shape}")
print(f"    deg([3]-isogeny) = {A.degree_of_isogeny_n(3)} = 3^{2*A.g}")

# ── 8. Theta divisor ──────────────────────────────────────────
print("\n[8] Theta divisor")
A1   = AbelianVariety(np.array([[1.3j]]), name="E")
TD   = ThetaDivisor(A1)
print(f"    {TD}")
print(f"    Riemann-Roch: {TD.riemann_roch()}")
print(f"    Poincaré:     {TD.poincare_line_bundle()}")
# Sample near-zero points
zeros = TD.divisor_points_1d(n_grid=20)
print(f"    Near-zero θ points found: {len(zeros)}")

print("\n[OK] Example 09 complete.")
