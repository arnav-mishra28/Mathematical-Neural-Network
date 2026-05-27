"""mnn.discovery.patterns — Pattern Discovery Engine.

Engine 2: Searches for symmetries, invariants, conserved quantities,
and recurring structures. Uses symbolic regression with sparsity
and symmetry penalties to discover governing equations from data.
"""
from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Any, Callable, Dict, List, Optional, Tuple
from itertools import combinations


class SymmetryDetector:
    """Detect symmetries in functions, sequences, and data."""

    @staticmethod
    def test_commutativity(op: Callable, elements: List[Any],
                           n_samples: int = 100) -> Dict:
        """Test if a binary operation is commutative: op(a,b) = op(b,a)."""
        violations = 0
        tested = 0
        rng = np.random.default_rng(42)
        indices = rng.integers(0, len(elements), size=(n_samples, 2))
        for i, j in indices:
            a, b = elements[i], elements[j]
            try:
                ab = op(a, b)
                ba = op(b, a)
                tested += 1
                if isinstance(ab, np.ndarray):
                    if not np.allclose(ab, ba, atol=1e-8):
                        violations += 1
                elif abs(ab - ba) > 1e-8:
                    violations += 1
            except Exception:
                pass
        rate = violations / max(tested, 1)
        return {"commutative": rate < 0.01, "violation_rate": rate, "tested": tested}

    @staticmethod
    def test_associativity(op: Callable, elements: List[Any],
                            n_samples: int = 50) -> Dict:
        """Test if op((a,b),c) = op(a,op(b,c))."""
        violations = 0
        tested = 0
        rng = np.random.default_rng(42)
        indices = rng.integers(0, len(elements), size=(n_samples, 3))
        for i, j, k in indices:
            a, b, c = elements[i], elements[j], elements[k]
            try:
                lhs = op(op(a, b), c)
                rhs = op(a, op(b, c))
                tested += 1
                if isinstance(lhs, np.ndarray):
                    if not np.allclose(lhs, rhs, atol=1e-8):
                        violations += 1
                elif abs(lhs - rhs) > 1e-8:
                    violations += 1
            except Exception:
                pass
        rate = violations / max(tested, 1)
        return {"associative": rate < 0.01, "violation_rate": rate, "tested": tested}

    @staticmethod
    def detect_function_symmetry(f: Callable, dim: int,
                                  n_samples: int = 200) -> Dict:
        """Detect if f is even, odd, or periodic."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((n_samples, dim))
        fx = np.array([f(xi) for xi in x])
        f_neg = np.array([f(-xi) for xi in x])

        even_err = float(np.mean(np.abs(fx - f_neg)))
        odd_err = float(np.mean(np.abs(fx + f_neg)))

        symmetries = {}
        if even_err < 1e-6:
            symmetries["even"] = True
        if odd_err < 1e-6:
            symmetries["odd"] = True

        # Periodicity detection via autocorrelation
        if dim == 1:
            t = np.linspace(0, 10, 500)
            vals = np.array([f(np.array([ti])) for ti in t])
            if isinstance(vals[0], (int, float, np.floating)):
                vals = vals - np.mean(vals)
                autocorr = np.correlate(vals, vals, mode="full")
                autocorr = autocorr[len(autocorr)//2:]
                autocorr /= autocorr[0] + 1e-15
                peaks = []
                for i in range(1, len(autocorr) - 1):
                    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.5:
                        peaks.append(i)
                if peaks:
                    period = t[peaks[0]] - t[0]
                    symmetries["periodic"] = True
                    symmetries["estimated_period"] = float(period)

        return symmetries

    @staticmethod
    def detect_invariant(f: Callable, transform: Callable,
                          data: np.ndarray) -> Dict:
        """Test if f(T(x)) = f(x) for a transformation T."""
        fx = np.array([f(x) for x in data])
        ftx = np.array([f(transform(x)) for x in data])
        err = float(np.mean(np.abs(fx - ftx)))
        return {"invariant": err < 1e-6, "error": err}


class InvariantFinder:
    """Search for conserved quantities in dynamical systems."""

    @staticmethod
    def find_conservation_laws(trajectory: np.ndarray,
                                candidate_fns: Optional[List[Callable]] = None,
                                tol: float = 1e-4) -> List[Dict]:
        """Find quantities that remain constant along a trajectory.

        Default candidates: norms, energies, angular momentum, etc.
        """
        if candidate_fns is None:
            candidate_fns = InvariantFinder._default_candidates(trajectory.shape[1])

        conserved = []
        for i, fn in enumerate(candidate_fns):
            try:
                values = np.array([fn(x) for x in trajectory])
                std = float(np.std(values))
                mean = float(np.mean(values))
                if std / (abs(mean) + 1e-15) < tol:
                    conserved.append({
                        "index": i, "name": fn.__name__ if hasattr(fn, '__name__') else f"Q_{i}",
                        "mean_value": mean, "std": std,
                        "relative_variation": std / (abs(mean) + 1e-15),
                    })
            except Exception:
                pass
        return conserved

    @staticmethod
    def _default_candidates(dim: int) -> List[Callable]:
        fns = []

        def norm_sq(x):
            return float(np.sum(x**2))
        norm_sq.__name__ = "||x||^2"
        fns.append(norm_sq)

        def norm(x):
            return float(np.linalg.norm(x))
        norm.__name__ = "||x||"
        fns.append(norm)

        for i in range(min(dim, 5)):
            def comp_sq(x, _i=i):
                return float(x[_i]**2)
            comp_sq.__name__ = f"x_{i}^2"
            fns.append(comp_sq)

        if dim >= 2:
            for i, j in combinations(range(min(dim, 4)), 2):
                def cross(x, _i=i, _j=j):
                    return float(x[_i] * x[_j])
                cross.__name__ = f"x_{i}*x_{j}"
                fns.append(cross)

            def angular_2d(x):
                return float(x[0]**2 + x[1]**2)
            angular_2d.__name__ = "x0^2+x1^2"
            fns.append(angular_2d)

        if dim >= 3:
            def angular_mom(x):
                return float(x[0]*x[4] - x[1]*x[3]) if dim >= 5 else float(x[0]*x[2])
            angular_mom.__name__ = "L_z"
            fns.append(angular_mom)

        return fns


class SequenceAnalyzer:
    """Discover patterns in numerical sequences."""

    @staticmethod
    def analyze(seq: List[float]) -> Dict:
        """Comprehensive sequence analysis."""
        arr = np.array(seq, dtype=float)
        n = len(arr)
        results = {"length": n, "values": seq}

        # Differences
        d1 = np.diff(arr)
        d2 = np.diff(d1) if len(d1) > 1 else np.array([])

        if np.allclose(d1, d1[0], atol=1e-8):
            results["pattern"] = "arithmetic"
            results["common_difference"] = float(d1[0])
            results["formula"] = f"a(n) = {arr[0]:.4g} + {d1[0]:.4g}*n"
        elif len(d2) > 0 and np.allclose(d2, d2[0], atol=1e-8):
            results["pattern"] = "quadratic"
            # Fit a + bn + cn^2
            ns = np.arange(n)
            coeffs = np.polyfit(ns, arr, 2)
            results["coefficients"] = coeffs.tolist()
            results["formula"] = f"a(n) = {coeffs[0]:.4g}*n^2 + {coeffs[1]:.4g}*n + {coeffs[2]:.4g}"
        else:
            # Check geometric
            ratios = arr[1:] / (arr[:-1] + 1e-15)
            if len(ratios) > 0 and np.allclose(ratios, ratios[0], atol=1e-8) and abs(ratios[0]) > 1e-8:
                results["pattern"] = "geometric"
                results["common_ratio"] = float(ratios[0])
                results["formula"] = f"a(n) = {arr[0]:.4g} * {ratios[0]:.4g}^n"
            else:
                # Polynomial fit attempt
                ns = np.arange(n)
                for deg in range(1, min(6, n-1)):
                    coeffs = np.polyfit(ns, arr, deg)
                    fitted = np.polyval(coeffs, ns)
                    if np.allclose(fitted, arr, atol=1e-6):
                        results["pattern"] = f"polynomial (degree {deg})"
                        results["coefficients"] = coeffs.tolist()
                        break
                else:
                    results["pattern"] = "unknown"

        return results

    @staticmethod
    def predict_next(seq: List[float], n_predict: int = 3) -> List[float]:
        """Predict next values based on discovered pattern."""
        info = SequenceAnalyzer.analyze(seq)
        n = len(seq)
        predictions = []

        if info["pattern"] == "arithmetic":
            d = info["common_difference"]
            for k in range(n_predict):
                predictions.append(seq[-1] + d * (k + 1))
        elif info["pattern"] == "geometric":
            r = info["common_ratio"]
            for k in range(n_predict):
                predictions.append(seq[-1] * r**(k + 1))
        elif "coefficients" in info:
            coeffs = info["coefficients"]
            for k in range(n_predict):
                predictions.append(float(np.polyval(coeffs, n + k)))
        else:
            predictions = [seq[-1]] * n_predict

        return predictions
