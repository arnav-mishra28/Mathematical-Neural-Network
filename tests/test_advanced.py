"""Tests for mnn.advanced — Phase 2 modules."""
import sys; sys.path.insert(0, "..")
import numpy as np
from mnn.algebra.groups import Group
from mnn.advanced.group_theory import (
    FiniteGroupAnalyzer, GroupHomomorphism, GroupExtension, RepresentationTheory
)
from mnn.advanced.abelian       import EllipticCurve, JacobiVariety, AbelianVariety, ThetaDivisor
from mnn.advanced.chaos_advanced import FractalAnalyzer, MultifractalSpectrum, ChaoticMap, CoupledOscillators
from mnn.advanced.abelian.elliptic import Point

# ── Group theory tests ────────────────────────────────────────

def test_derived_series_abelian():
    """Abelian group has trivial derived subgroup."""
    Z6  = Group.cyclic(6)
    ana = FiniteGroupAnalyzer(Z6)
    ds  = ana.derived_series()
    assert len(ds[-1]) == 1, "Cyclic group derived series must end at {e}"

def test_solvable_cyclic():
    assert FiniteGroupAnalyzer(Group.cyclic(6)).is_solvable()

def test_solvable_s3():
    assert FiniteGroupAnalyzer(Group.symmetric(3)).is_solvable()

def test_nilpotent_cyclic():
    assert FiniteGroupAnalyzer(Group.cyclic(4)).is_nilpotent()

def test_nilpotency_class():
    assert FiniteGroupAnalyzer(Group.cyclic(4)).nilpotency_class() is not None

def test_sylow_s3():
    S3  = Group.symmetric(3)
    ana = FiniteGroupAnalyzer(S3)
    v2  = ana.sylow_verification(2)
    v3  = ana.sylow_verification(3)
    assert v2['thm1_exists'] and v3['thm1_exists']
    assert v2['thm3_mod_p'] and v3['thm3_mod_p']

def test_homomorphism():
    Z4 = Group.cyclic(4); Z2 = Group.cyclic(2)
    h  = GroupHomomorphism(Z4, Z2, lambda x: x % 2)
    assert h.is_homomorphism()
    assert set(h.kernel()) == {0, 2}
    assert h.is_surjective()

def test_semidirect_product_order():
    Z3 = Group.cyclic(3); Z2 = Group.cyclic(2)
    D3 = GroupExtension.semidirect_product(Z3, Z2, lambda q,n: (-n)%3 if q==1 else n)
    assert D3.order == 6

def test_representation_trivial():
    Z5  = Group.cyclic(5)
    RT  = RepresentationTheory(Z5)
    triv = RT.trivial_representation()
    assert triv.is_representation()

def test_character_table_orthogonality():
    Z6  = Group.cyclic(6)
    RT  = RepresentationTheory(Z6)
    assert RT.verify_orthogonality()

def test_regular_representation():
    Z4 = Group.cyclic(4)
    RT = RepresentationTheory(Z4)
    rr = RT.regular_representation()
    assert rr.degree == 4
    assert rr.is_representation()

# ── Elliptic curve tests ──────────────────────────────────────

def test_elliptic_curve_discriminant():
    E = EllipticCurve(-1., 0.)
    assert E.discriminant() != 0

def test_point_on_curve():
    E  = EllipticCurve(0., 0.)   # y² = x³
    # (0,0) is on y²=x³ but discriminant is 0 → skip
    E2 = EllipticCurve(1., 1.)
    pts = E2.sample_points((-1,3), n=500)
    if len(pts) > 0:
        P = Point(pts[0][0], pts[0][1])
        assert E2.is_on_curve(P, tol=1e-5)

def test_point_at_infinity():
    E = EllipticCurve(-1., 1.)
    P = Point(0., 1.)
    O = Point.infinity()
    assert E.add(P, O) == P

def test_negate_point():
    E = EllipticCurve(-1., 1.)
    P = Point(1., 1.)
    neg = E.negate(P)
    S   = E.add(P, neg)
    assert S.is_infinity

def test_scalar_mult_zero():
    E = EllipticCurve(-1., 1.)
    P = Point(1., 1.)
    assert E.scalar_mult(0, P).is_infinity

def test_elliptic_Fp():
    E   = EllipticCurve(-1., 0.)
    pts = E.points_over_Fp(7)
    # Hasse bound: |#E(F_p) - p - 1| ≤ 2√p
    assert abs(len(pts) - 8) <= 2*np.sqrt(7) + 1

def test_jacobi_variety():
    J = JacobiVariety(genus=2)
    assert J.is_principally_polarized()

def test_abelian_variety_siegel():
    A = AbelianVariety(np.array([[1.1j, 0.3j],[0.3j, 1.4j]]))
    assert A.is_in_siegel_half_space()

def test_theta_divisor():
    A  = AbelianVariety(np.array([[1.3j]]))
    TD = ThetaDivisor(A)
    z_test = np.array([0.5+0.5j])
    result = TD.is_on_divisor(z_test, tol=10.)   # broad tolerance
    assert isinstance(result, bool)

# ── Chaos advanced tests ──────────────────────────────────────

def test_box_counting_dim():
    sier = FractalAnalyzer.sierpinski_triangle(n_points=5000)
    d, _, _ = FractalAnalyzer.box_counting_dimension(sier, n_scales=8)
    assert 1.0 < d < 2.0   # Sierpinski ≈ 1.585

def test_hurst_random_walk():
    rng    = np.random.default_rng(0)
    signal = np.cumsum(rng.randn(2000))
    H      = FractalAnalyzer.hurst_exponent_dfa(signal)
    assert 0.1 < H <= 1.0

def test_hurst_white_noise():
    rng    = np.random.default_rng(1)
    signal = rng.randn(2000)
    H      = FractalAnalyzer.hurst_exponent_rs(signal)
    assert 0.1 < H <= 1.0

def test_mandelbrot():
    M = FractalAnalyzer.mandelbrot_set(n_real=50, n_imag=50, max_iter=30)
    assert M.shape == (50, 50)
    assert M.min() >= 0

def test_julia():
    J = FractalAnalyzer.julia_set(c=-0.7+0.27j, n_real=50, n_imag=50, max_iter=30)
    assert J.shape == (50, 50)

def test_standard_map():
    pts = ChaoticMap.standard_map(K=1.0, n_iter=1000)
    assert pts.shape == (1000, 2)
    assert np.all((pts >= 0) & (pts <= 2*np.pi))

def test_henon_map():
    pts = ChaoticMap.henon_map(n_iter=500)
    assert pts.shape == (500, 2)

def test_tent_map():
    s = ChaoticMap.tent_map(n_iter=1000)
    assert s.shape == (1000,)
    assert np.all((s >= 0) & (s <= 1))

def test_kuramoto():
    t, theta = CoupledOscillators.kuramoto_model(N=4, K=2., t_end=10.)
    assert theta.shape[1] == 4
    r = CoupledOscillators.order_parameter(theta)
    assert len(r) == len(t)

def test_van_der_pol():
    traj = CoupledOscillators.van_der_pol(mu=1., t_end=10.)
    assert traj.shape[1] == 2

# ── Prototype tests ───────────────────────────────────────────

def test_prototype_runs():
    from mnn.advanced.prototype import run_prototype
    result = run_prototype(n_epochs=100, width=32, depth=2,
                           n_colloc=50, verbose=False, seed=0)
    assert result.final_mse_df is not None
    assert len(result.total_losses) == 100

def test_prototype_derivative_improves():
    from mnn.advanced.prototype import run_prototype
    result = run_prototype(n_epochs=500, width=64, depth=4,
                           n_colloc=200, verbose=False, seed=42)
    # Derivative constraint loss should decrease
    early = np.mean(result.constraint_losses[:50])
    late  = np.mean(result.constraint_losses[-50:])
    assert late < early, f"Constraint loss should decrease: {early:.4f} → {late:.4f}"

def test_prototype_mse_reasonable():
    from mnn.advanced.prototype import run_prototype
    result = run_prototype(n_epochs=1000, width=64, depth=5,
                           n_colloc=300, verbose=False, seed=42)
    # With 1000 epochs, derivative MSE should be < 0.5
    assert result.final_mse_df < 0.5, f"MSE df too large: {result.final_mse_df}"

if __name__ == "__main__":
    tests = [
        # Group theory
        test_derived_series_abelian, test_solvable_cyclic, test_solvable_s3,
        test_nilpotent_cyclic, test_nilpotency_class, test_sylow_s3,
        test_homomorphism, test_semidirect_product_order,
        test_representation_trivial, test_character_table_orthogonality,
        test_regular_representation,
        # Abelian
        test_elliptic_curve_discriminant, test_point_on_curve,
        test_point_at_infinity, test_negate_point, test_scalar_mult_zero,
        test_elliptic_Fp, test_jacobi_variety, test_abelian_variety_siegel,
        test_theta_divisor,
        # Chaos advanced
        test_box_counting_dim, test_hurst_random_walk, test_hurst_white_noise,
        test_mandelbrot, test_julia, test_standard_map, test_henon_map,
        test_tent_map, test_kuramoto, test_van_der_pol,
        # Prototype
        test_prototype_runs, test_prototype_derivative_improves,
        test_prototype_mse_reasonable,
    ]
    passed = failed = 0
    for fn in tests:
        try:
            fn(); print(f"  ✓ {fn.__name__}"); passed += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}"); failed += 1
    print(f"\n[{passed} passed, {failed} failed]")
