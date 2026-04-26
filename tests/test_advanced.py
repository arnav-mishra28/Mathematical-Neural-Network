"""Tests for mnn.advanced Phase 2 modules."""
import sys

import numpy as np

sys.path.insert(0, "..")

from mnn.algebra.groups import Group
from mnn.advanced.abelian import AbelianVariety, EllipticCurve, JacobiVariety, ThetaDivisor
from mnn.advanced.abelian.elliptic import Point
from mnn.advanced.chaos_advanced import (
    ChaoticMap,
    CoupledOscillators,
    FractalAnalyzer,
    MultifractalSpectrum,
)
from mnn.advanced.group_theory import (
    FiniteGroupAnalyzer,
    GroupExtension,
    GroupHomomorphism,
    RepresentationTheory,
)


def test_derived_series_abelian():
    """Abelian group has trivial derived subgroup."""
    z6 = Group.cyclic(6)
    analyzer = FiniteGroupAnalyzer(z6)
    derived = analyzer.derived_series()
    assert len(derived[-1]) == 1, "Cyclic group derived series must end at {e}"


def test_solvable_cyclic():
    assert FiniteGroupAnalyzer(Group.cyclic(6)).is_solvable()


def test_solvable_s3():
    assert FiniteGroupAnalyzer(Group.symmetric(3)).is_solvable()


def test_nilpotent_cyclic():
    assert FiniteGroupAnalyzer(Group.cyclic(4)).is_nilpotent()


def test_nilpotency_class():
    assert FiniteGroupAnalyzer(Group.cyclic(4)).nilpotency_class() is not None


def test_sylow_s3():
    s3 = Group.symmetric(3)
    analyzer = FiniteGroupAnalyzer(s3)
    v2 = analyzer.sylow_verification(2)
    v3 = analyzer.sylow_verification(3)
    assert v2["thm1_exists"] and v3["thm1_exists"]
    assert v2["thm3_mod_p"] and v3["thm3_mod_p"]


def test_homomorphism():
    z4 = Group.cyclic(4)
    z2 = Group.cyclic(2)
    hom = GroupHomomorphism(z4, z2, lambda x: x % 2)
    assert hom.is_homomorphism()
    assert set(hom.kernel()) == {0, 2}
    assert hom.is_surjective()


def test_semidirect_product_order():
    z3 = Group.cyclic(3)
    z2 = Group.cyclic(2)
    d3 = GroupExtension.semidirect_product(z3, z2, lambda q, n: (-n) % 3 if q == 1 else n)
    assert d3.order == 6


def test_representation_trivial():
    z5 = Group.cyclic(5)
    theory = RepresentationTheory(z5)
    trivial = theory.trivial_representation()
    assert trivial.is_representation()


def test_character_table_orthogonality():
    z6 = Group.cyclic(6)
    theory = RepresentationTheory(z6)
    assert theory.verify_orthogonality()


def test_regular_representation():
    z4 = Group.cyclic(4)
    theory = RepresentationTheory(z4)
    regular = theory.regular_representation()
    assert regular.degree == 4
    assert regular.is_representation()


def test_elliptic_curve_discriminant():
    curve = EllipticCurve(-1.0, 0.0)
    assert curve.discriminant() != 0


def test_point_on_curve():
    curve = EllipticCurve(1.0, 1.0)
    points = curve.sample_points((-1, 3), n=500)
    if len(points) > 0:
        point = Point(points[0][0], points[0][1])
        assert curve.is_on_curve(point, tol=1e-5)


def test_point_at_infinity():
    curve = EllipticCurve(-1.0, 1.0)
    point = Point(0.0, 1.0)
    infinity = Point.infinity()
    assert curve.add(point, infinity) == point


def test_negate_point():
    curve = EllipticCurve(-1.0, 1.0)
    point = Point(1.0, 1.0)
    negated = curve.negate(point)
    summed = curve.add(point, negated)
    assert summed.is_infinity


def test_scalar_mult_zero():
    curve = EllipticCurve(-1.0, 1.0)
    point = Point(1.0, 1.0)
    assert curve.scalar_mult(0, point).is_infinity


def test_elliptic_Fp():
    curve = EllipticCurve(-1.0, 0.0)
    points = curve.points_over_Fp(7)
    assert abs(len(points) - 8) <= 2 * np.sqrt(7) + 1


def test_jacobi_variety():
    jacobian = JacobiVariety(genus=2)
    assert jacobian.is_principally_polarized()


def test_abelian_variety_siegel():
    variety = AbelianVariety(np.array([[1.1j, 0.3j], [0.3j, 1.4j]]))
    assert variety.is_in_siegel_half_space()


def test_theta_divisor():
    variety = AbelianVariety(np.array([[1.3j]]))
    divisor = ThetaDivisor(variety)
    z_test = np.array([0.5 + 0.5j])
    result = divisor.is_on_divisor(z_test, tol=10.0)
    assert isinstance(result, bool)


def test_box_counting_dim():
    sierpinski = FractalAnalyzer.sierpinski_triangle(n_points=5000)
    dimension, _, _ = FractalAnalyzer.box_counting_dimension(sierpinski, n_scales=8)
    assert 1.0 < dimension < 2.0


def test_hurst_random_walk():
    rng = np.random.default_rng(0)
    signal = np.cumsum(rng.standard_normal(2000))
    hurst = FractalAnalyzer.hurst_exponent_dfa(signal)
    assert 0.1 < hurst <= 1.0


def test_hurst_white_noise():
    rng = np.random.default_rng(1)
    signal = rng.standard_normal(2000)
    hurst = FractalAnalyzer.hurst_exponent_rs(signal)
    assert 0.1 < hurst <= 1.0


def test_mandelbrot():
    values = FractalAnalyzer.mandelbrot_set(n_real=50, n_imag=50, max_iter=30)
    assert values.shape == (50, 50)
    assert values.min() >= 0


def test_julia():
    values = FractalAnalyzer.julia_set(c=-0.7 + 0.27j, n_real=50, n_imag=50, max_iter=30)
    assert values.shape == (50, 50)


def test_standard_map():
    points = ChaoticMap.standard_map(K=1.0, n_iter=1000)
    assert points.shape == (1000, 2)
    assert np.all((points >= 0) & (points <= 2 * np.pi))


def test_henon_map():
    points = ChaoticMap.henon_map(n_iter=500)
    assert points.shape == (500, 2)


def test_tent_map():
    series = ChaoticMap.tent_map(n_iter=1000)
    assert series.shape == (1000,)
    assert np.all((series >= 0) & (series <= 1))


def test_kuramoto():
    time, theta = CoupledOscillators.kuramoto_model(N=4, K=2.0, t_end=10.0)
    assert theta.shape[1] == 4
    order_parameter = CoupledOscillators.order_parameter(theta)
    assert len(order_parameter) == len(time)


def test_van_der_pol():
    trajectory = CoupledOscillators.van_der_pol(mu=1.0, t_end=10.0)
    assert trajectory.shape[1] == 2


def test_prototype_runs():
    from mnn.advanced.prototype import run_prototype

    result = run_prototype(n_epochs=100, width=32, depth=2, n_colloc=50, verbose=False, seed=0)
    assert result.final_mse_df is not None
    assert len(result.total_losses) == 100


def test_prototype_derivative_improves():
    from mnn.advanced.prototype import run_prototype

    result = run_prototype(n_epochs=500, width=64, depth=4, n_colloc=200, verbose=False, seed=42)
    early = np.mean(result.constraint_losses[:50])
    late = np.mean(result.constraint_losses[-50:])
    assert late < early, f"Constraint loss should decrease: {early:.4f} -> {late:.4f}"


def test_prototype_mse_reasonable():
    from mnn.advanced.prototype import run_prototype

    result = run_prototype(n_epochs=1000, width=64, depth=5, n_colloc=300, verbose=False, seed=42)
    assert result.final_mse_df < 0.5, f"MSE df too large: {result.final_mse_df}"


if __name__ == "__main__":
    tests = [
        test_derived_series_abelian,
        test_solvable_cyclic,
        test_solvable_s3,
        test_nilpotent_cyclic,
        test_nilpotency_class,
        test_sylow_s3,
        test_homomorphism,
        test_semidirect_product_order,
        test_representation_trivial,
        test_character_table_orthogonality,
        test_regular_representation,
        test_elliptic_curve_discriminant,
        test_point_on_curve,
        test_point_at_infinity,
        test_negate_point,
        test_scalar_mult_zero,
        test_elliptic_Fp,
        test_jacobi_variety,
        test_abelian_variety_siegel,
        test_theta_divisor,
        test_box_counting_dim,
        test_hurst_random_walk,
        test_hurst_white_noise,
        test_mandelbrot,
        test_julia,
        test_standard_map,
        test_henon_map,
        test_tent_map,
        test_kuramoto,
        test_van_der_pol,
        test_prototype_runs,
        test_prototype_derivative_improves,
        test_prototype_mse_reasonable,
    ]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  fail {fn.__name__}: {exc}")
            failed += 1
    print(f"\n[{passed} passed, {failed} failed]")
