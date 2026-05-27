"""mnn.discovery.validation — Proof / Validation Engine.

Engine 4: Validates conjectures via numerical verification, symbolic
consistency checks, and counterexample search. First-level validation
before formal proving.
"""
from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Any, Callable, Dict, List, Optional, Tuple
from .conjectures import Conjecture, ConjectureStatus


class NumericalValidator:
    """Validate conjectures numerically across diverse inputs."""

    @staticmethod
    def validate_equality(lhs: Callable, rhs: Callable,
                           domain: np.ndarray,
                           atol: float = 1e-8) -> Dict:
        """Test lhs(x) == rhs(x) across domain."""
        n_pass = 0
        n_fail = 0
        max_err = 0.0
        failures = []

        for x in domain:
            try:
                l = lhs(x)
                r = rhs(x)
                err = abs(l - r) if np.isscalar(l) else float(np.max(np.abs(np.array(l) - np.array(r))))
                max_err = max(max_err, err)
                if err < atol:
                    n_pass += 1
                else:
                    n_fail += 1
                    if len(failures) < 5:
                        failures.append({"input": x, "lhs": l, "rhs": r, "error": err})
            except Exception as e:
                n_fail += 1
                if len(failures) < 5:
                    failures.append({"input": x, "error": str(e)})

        total = n_pass + n_fail
        return {
            "valid": n_fail == 0,
            "pass_rate": n_pass / max(total, 1),
            "n_tested": total,
            "max_error": max_err,
            "failures": failures,
        }

    @staticmethod
    def validate_inequality(lhs: Callable, rhs: Callable,
                             domain: np.ndarray) -> Dict:
        """Test lhs(x) <= rhs(x) across domain."""
        violations = []
        for x in domain:
            try:
                l, r = lhs(x), rhs(x)
                if l > r + 1e-10:
                    violations.append({"input": x, "lhs": l, "rhs": r})
            except Exception:
                pass
        return {"valid": len(violations) == 0, "violations": violations[:5]}

    @staticmethod
    def validate_property(prop_fn: Callable, domain: np.ndarray,
                           description: str = "") -> Dict:
        """Test if prop_fn(x) returns True for all x in domain."""
        n_pass = 0
        n_fail = 0
        failures = []
        for x in domain:
            try:
                if prop_fn(x):
                    n_pass += 1
                else:
                    n_fail += 1
                    if len(failures) < 5:
                        failures.append({"input": x})
            except Exception:
                n_fail += 1
        return {
            "valid": n_fail == 0,
            "pass_rate": n_pass / max(n_pass + n_fail, 1),
            "description": description,
            "failures": failures,
        }


class CounterexampleSearcher:
    """Search for counterexamples to refute conjectures."""

    @staticmethod
    def random_search(prop_fn: Callable, dim: int,
                       n_samples: int = 10000,
                       ranges: Optional[Tuple[float, float]] = None,
                       seed: int = 42) -> Dict:
        """Random search for counterexamples."""
        rng = np.random.default_rng(seed)
        lo, hi = ranges or (-10, 10)

        for i in range(n_samples):
            x = rng.uniform(lo, hi, size=dim)
            try:
                if not prop_fn(x):
                    return {
                        "found": True,
                        "counterexample": x.tolist(),
                        "iteration": i,
                    }
            except Exception:
                pass

        return {"found": False, "n_searched": n_samples}

    @staticmethod
    def boundary_search(prop_fn: Callable, dim: int,
                         boundaries: Optional[List[float]] = None) -> Dict:
        """Test at boundary/special values."""
        boundaries = boundaries or [0, 1, -1, 0.5, -0.5, 2, -2, 10, -10,
                                     np.pi, -np.pi, np.e, np.sqrt(2)]
        counterexamples = []

        for b in boundaries:
            x = np.full(dim, b)
            try:
                if not prop_fn(x):
                    counterexamples.append(x.tolist())
            except Exception:
                pass

        # Also test zero vector, ones, etc.
        for special in [np.zeros(dim), np.ones(dim), -np.ones(dim)]:
            try:
                if not prop_fn(special):
                    counterexamples.append(special.tolist())
            except Exception:
                pass

        return {"found": len(counterexamples) > 0, "counterexamples": counterexamples[:5]}

    @staticmethod
    def gradient_search(loss_fn: Callable, dim: int,
                         n_starts: int = 10, n_steps: int = 100,
                         lr: float = 0.1) -> Dict:
        """Gradient-based search to maximize violation."""
        import torch
        best_violation = 0.0
        best_x = None

        for _ in range(n_starts):
            x = torch.randn(dim, requires_grad=True)
            optimizer = torch.optim.Adam([x], lr=lr)
            for _ in range(n_steps):
                optimizer.zero_grad()
                loss = -loss_fn(x)  # maximize violation
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                violation = loss_fn(x).item()
                if violation > best_violation:
                    best_violation = violation
                    best_x = x.detach().numpy().tolist()

        return {
            "found": best_violation > 1e-4,
            "max_violation": best_violation,
            "worst_input": best_x,
        }


class SymbolicValidator:
    """Validate conjectures using symbolic mathematics."""

    @staticmethod
    def verify_identity(lhs_expr: sp.Expr, rhs_expr: sp.Expr,
                         variables: List[str]) -> Dict:
        """Check if lhs == rhs symbolically."""
        diff = sp.simplify(lhs_expr - rhs_expr)
        is_zero = diff == 0 or sp.simplify(diff).is_zero
        return {
            "proven": bool(is_zero),
            "simplified_difference": str(diff),
            "method": "symbolic_simplification",
        }

    @staticmethod
    def verify_inequality(lhs_expr: sp.Expr, rhs_expr: sp.Expr) -> Dict:
        """Attempt symbolic verification of lhs <= rhs."""
        diff = sp.simplify(rhs_expr - lhs_expr)
        # Check if diff is always non-negative
        try:
            is_nonneg = sp.ask(sp.Q.nonnegative(diff))
            return {
                "proven": bool(is_nonneg),
                "difference": str(diff),
                "method": "symbolic_assumption",
            }
        except Exception:
            return {"proven": False, "difference": str(diff), "method": "failed"}

    @staticmethod
    def check_dimensional_consistency(expr: sp.Expr,
                                        dimensions: Dict[str, str]) -> Dict:
        """Check dimensional consistency of an expression.

        dimensions: map variable name -> dimension string (e.g. "L", "T", "M")
        """
        # Simple structural check — all terms in a sum must have same dimension
        if isinstance(expr, sp.Add):
            term_dims = []
            for term in expr.args:
                symbols_in_term = term.free_symbols
                dims = {str(s): dimensions.get(str(s), "?") for s in symbols_in_term}
                term_dims.append(dims)
            consistent = all(d == term_dims[0] for d in term_dims) if term_dims else True
            return {"consistent": consistent, "term_dimensions": term_dims}
        return {"consistent": True, "note": "single term or non-additive"}


class ConjectureValidator:
    """End-to-end conjecture validation pipeline."""

    def __init__(self):
        self.numerical = NumericalValidator()
        self.counterexample = CounterexampleSearcher()
        self.symbolic = SymbolicValidator()

    def validate_conjecture(self, conjecture: Conjecture,
                             test_fn: Optional[Callable] = None,
                             domain: Optional[np.ndarray] = None,
                             dim: int = 1,
                             n_numerical: int = 1000,
                             n_counter: int = 5000) -> Conjecture:
        """Full validation pipeline for a conjecture."""
        if test_fn is not None and domain is not None:
            # Numerical validation
            result = self.numerical.validate_property(test_fn, domain)
            if result["valid"]:
                conjecture.add_evidence(
                    f"Passed numerical validation ({result['pass_rate']:.0%} of {len(domain)} tests)")
            else:
                conjecture.add_counterexample(
                    f"Failed numerical validation (pass rate: {result['pass_rate']:.0%})",
                    result["failures"]
                )
                return conjecture

        if test_fn is not None:
            # Counterexample search
            cx = self.counterexample.random_search(test_fn, dim, n_counter)
            if cx["found"]:
                conjecture.add_counterexample(
                    f"Counterexample found at iteration {cx.get('iteration', '?')}",
                    cx["counterexample"]
                )
            else:
                conjecture.add_evidence(
                    f"No counterexample found in {cx['n_searched']} random samples")

            # Boundary search
            bx = self.counterexample.boundary_search(test_fn, dim)
            if bx["found"]:
                conjecture.add_counterexample(
                    "Counterexample at boundary value", bx["counterexamples"]
                )
            else:
                conjecture.add_evidence("Passed boundary value checks")

        return conjecture
