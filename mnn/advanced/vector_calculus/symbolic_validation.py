"""
mnn.advanced.vector_calculus.symbolic_validation
==================================================
SymPy-based symbolic validation of learned fields.

This module cross-validates the neural network outputs against
exact symbolic solutions — enabling Wolfram Mathematica-level
verification of what the network learned.

Key capabilities:
  - Define exact target fields symbolically
  - Compute exact operators symbolically (∇, ∇·, ∇×, ∇²)
  - Compare network outputs numerically to symbolic ground truth
  - Generate symbolic residual expressions for constraint verification
  - Export learned fields as symbolic approximations (Taylor series)
  - Produce LaTeX reports of all operators and residuals
"""

from __future__ import annotations
import numpy as np
import sympy as sp
import torch
import torch.nn as nn
from typing import List, Dict, Callable, Optional, Tuple, Union


class SymbolicField:
    """
    A field defined symbolically via SymPy.
    Supports automatic computation of all differential operators.
    """

    def __init__(self, expressions: Union[sp.Expr, List[sp.Expr]],
                 variables: List[str],
                 name: str = "F"):
        self.variables   = [sp.Symbol(v) for v in variables]
        self.var_names   = variables
        self.name        = name
        self.dim         = len(variables)

        if isinstance(expressions, (sp.Expr, sp.core.basic.Basic)):
            self.components = [expressions]   # scalar field
            self.is_scalar  = True
        else:
            self.components = list(expressions)
            self.is_scalar  = len(self.components) == 1

    # ── Symbolic operators ─────────────────────────────────────────────────

    def gradient(self) -> "SymbolicField":
        """∇f — gradient of a scalar field."""
        if not self.is_scalar:
            raise ValueError("Gradient defined for scalar fields only.")
        grad = [sp.diff(self.components[0], v) for v in self.variables]
        return SymbolicField(grad, self.var_names, name=f"∇{self.name}")

    def divergence(self) -> "SymbolicField":
        """∇·F — divergence of a vector field."""
        if self.is_scalar:
            raise ValueError("Divergence defined for vector fields only.")
        if len(self.components) != self.dim:
            raise ValueError("dim(F) must equal dim(space) for divergence")
        div = sum(sp.diff(c, v) for c, v in zip(self.components, self.variables))
        return SymbolicField(sp.simplify(div), self.var_names, name=f"∇·{self.name}")

    def curl(self) -> "SymbolicField":
        """∇×F — curl of a 3D vector field."""
        if self.dim != 3 or len(self.components) != 3:
            raise ValueError("Curl requires 3D vector field")
        Fx, Fy, Fz = self.components; x, y, z = self.variables
        cx = sp.simplify(sp.diff(Fz,y) - sp.diff(Fy,z))
        cy = sp.simplify(sp.diff(Fx,z) - sp.diff(Fz,x))
        cz = sp.simplify(sp.diff(Fy,x) - sp.diff(Fx,y))
        return SymbolicField([cx,cy,cz], self.var_names, name=f"∇×{self.name}")

    def laplacian(self) -> "SymbolicField":
        """∇²f — Laplacian of a scalar or vector field."""
        if self.is_scalar:
            lap = sp.simplify(sum(sp.diff(self.components[0], v, 2) for v in self.variables))
            return SymbolicField(lap, self.var_names, name=f"∇²{self.name}")
        # Vector Laplacian: apply component-wise
        lap_comps = []
        for c in self.components:
            lap_comps.append(sp.simplify(sum(sp.diff(c, v, 2) for v in self.variables)))
        return SymbolicField(lap_comps, self.var_names, name=f"∇²{self.name}")

    def jacobian(self) -> sp.Matrix:
        """J_ij = ∂Fᵢ/∂xⱼ — Jacobian matrix."""
        return sp.Matrix([[sp.diff(c, v) for v in self.variables]
                          for c in self.components])

    def hessian(self) -> sp.Matrix:
        """Hessian of a scalar field."""
        if not self.is_scalar: raise ValueError("Hessian for scalar fields only")
        return sp.hessian(self.components[0], self.variables)

    # ── Numerical evaluation ───────────────────────────────────────────────

    def evaluate_numpy(self, points: np.ndarray) -> np.ndarray:
        """Evaluate field at numpy array of points. Returns (N, n_components)."""
        funcs = [sp.lambdify(self.variables, c, modules=["numpy"])
                 for c in self.components]
        N = len(points)
        out = np.zeros((N, len(self.components)))
        for k, fn in enumerate(funcs):
            args = [points[:, i] for i in range(self.dim)]
            out[:, k] = fn(*args)
        return out

    def to_latex(self) -> str:
        if self.is_scalar:
            return sp.latex(self.components[0])
        inner = r", \\ ".join(sp.latex(c) for c in self.components)
        return r"\begin{pmatrix}" + inner + r"\end{pmatrix}"

    def __repr__(self):
        return f"SymbolicField({self.name}, vars={self.var_names}, scalar={self.is_scalar})"


class SymbolicValidator:
    """
    Cross-validates neural field networks against symbolic ground truth.

    Workflow:
      1. Define exact field symbolically via SymPy
      2. Train MNN network
      3. Validate: compare outputs, verify operators, compute error norms
      4. Generate LaTeX report
    """

    def __init__(self, space_dim: int = 3, variables: Optional[List[str]] = None):
        self.dim       = space_dim
        self.var_names = variables or ["x","y","z"][:space_dim]

    # ── Pre-built exact fields ─────────────────────────────────────────────

    def harmonic_field_2d(self) -> SymbolicField:
        """f(x,y) = sin(πx)sinh(πy) — exact harmonic function (∇²f=0)."""
        x, y = sp.symbols(" ".join(self.var_names[:2]))
        return SymbolicField(sp.sin(sp.pi*x)*sp.sinh(sp.pi*y), self.var_names[:2], "f_harmonic")

    def exact_gradient_field(self) -> SymbolicField:
        """F(x,y,z) = ∇(x²+y²+z²) = (2x, 2y, 2z). Curl-free, div=6."""
        syms = sp.symbols(" ".join(self.var_names))
        comps = [2*s for s in (syms if isinstance(syms, tuple) else (syms,))]
        return SymbolicField(comps, self.var_names, "∇r²")

    def exact_div_free_field_2d(self) -> SymbolicField:
        """F = (-y, x): pure rotation, ∇·F = 0, ∇×F = 2."""
        x, y = sp.symbols(self.var_names[0] + " " + self.var_names[1])
        return SymbolicField([-y, x], self.var_names[:2], "F_rot")

    def exact_div_free_field_3d(self) -> SymbolicField:
        """F = (y-z, z-x, x-y): ∇·F = 0 exactly."""
        x,y,z = sp.symbols(" ".join(self.var_names[:3]))
        return SymbolicField([y-z, z-x, x-y], self.var_names[:3], "F_div0")

    def taylor_green_vortex(self) -> SymbolicField:
        """Taylor-Green vortex: F = (sin x cos y, -cos x sin y, 0). ∇·F=0."""
        x,y,z = sp.symbols(" ".join(self.var_names[:3]))
        return SymbolicField([sp.sin(x)*sp.cos(y), -sp.cos(x)*sp.sin(y), sp.Integer(0)],
                              self.var_names[:3], "TaylorGreen")

    # ── Validation ─────────────────────────────────────────────────────────

    def validate_scalar_field(self, net: nn.Module,
                               exact: SymbolicField,
                               test_points: np.ndarray) -> Dict:
        """
        Compare scalar network output to exact symbolic field.
        Returns dict with max_error, mean_error, relative_error.
        """
        # Exact values
        exact_vals = exact.evaluate_numpy(test_points)[:, 0]   # (N,)

        # Network values
        net.eval()
        with torch.no_grad():
            xt      = torch.tensor(test_points, dtype=torch.float32)
            net_out = net(xt).squeeze(-1).numpy()

        # Align constant (network may learn f + C)
        offset  = float(np.mean(net_out - exact_vals))
        aligned = net_out - offset

        err = np.abs(aligned - exact_vals)
        return {
            "max_error":      float(err.max()),
            "mean_error":     float(err.mean()),
            "rms_error":      float(np.sqrt(np.mean(err**2))),
            "relative_error": float(err.mean() / (np.abs(exact_vals).mean() + 1e-15)),
            "offset":         offset,
        }

    def validate_vector_field(self, net: nn.Module,
                               exact: SymbolicField,
                               test_points: np.ndarray) -> Dict:
        """Compare vector network output to exact symbolic vector field."""
        exact_vals = exact.evaluate_numpy(test_points)   # (N, m)
        net.eval()
        with torch.no_grad():
            xt      = torch.tensor(test_points, dtype=torch.float32)
            net_out = net(xt).numpy()                     # (N, m)

        err = np.abs(net_out - exact_vals)
        return {
            "max_error":     float(err.max()),
            "mean_error":    float(err.mean()),
            "rms_error":     float(np.sqrt(np.mean(err**2))),
            "per_component": [float(np.sqrt(np.mean(err[:, i]**2))) for i in range(err.shape[1])],
        }

    def validate_constraint_symbolically(self, exact: SymbolicField,
                                          constraint: str) -> Dict:
        """
        Verify that the EXACT field satisfies a constraint symbolically.
        Returns dict with residual expression and whether it's zero.

        constraint : 'divergence_free' | 'curl_free' | 'harmonic'
        """
        result = {}
        if constraint == "divergence_free":
            div = exact.divergence()
            res = sp.simplify(div.components[0])
            result["residual_expr"] = res
            result["is_zero"]       = res == 0
            result["latex"]         = f"∇·{exact.name} = {sp.latex(res)}"
        elif constraint == "curl_free":
            curl = exact.curl()
            res  = [sp.simplify(c) for c in curl.components]
            result["residual_expr"] = res
            result["is_zero"]       = all(c == 0 for c in res)
            result["latex"]         = f"∇×{exact.name} = {[sp.latex(c) for c in res]}"
        elif constraint == "harmonic":
            lap  = exact.laplacian()
            res  = sp.simplify(lap.components[0])
            result["residual_expr"] = res
            result["is_zero"]       = res == 0
            result["latex"]         = f"∇²{exact.name} = {sp.latex(res)}"
        return result

    def operator_comparison(self, net: nn.Module,
                              exact: SymbolicField,
                              test_points: np.ndarray,
                              operator: str = "gradient") -> Dict:
        """
        Compare a differential operator applied to the network
        vs the same operator applied to the exact symbolic field.

        operator : 'gradient' | 'laplacian' | 'divergence' | 'curl'
        """
        from .operators import FieldOperators

        # Symbolic ground truth
        if operator == "gradient":
            exact_op = exact.gradient()
        elif operator == "laplacian":
            exact_op = exact.laplacian()
        elif operator == "divergence":
            exact_op = exact.divergence()
        elif operator == "curl":
            exact_op = exact.curl()
        else:
            raise ValueError(f"Unknown operator: {operator}")

        exact_vals = exact_op.evaluate_numpy(test_points)

        # Neural computation
        xt  = torch.tensor(test_points, dtype=torch.float32, requires_grad=True)
        net.eval()
        if operator == "gradient":
            net_out = FieldOperators.gradient(net, xt.clone(), create_graph=False).detach().numpy()
        elif operator == "laplacian":
            net_out = FieldOperators.laplacian(net, xt.clone(), create_graph=False).detach().numpy()
        elif operator == "divergence":
            net_out = FieldOperators.divergence(net, xt.clone(), create_graph=False).detach().numpy()
        elif operator == "curl":
            net_out = FieldOperators.curl(net, xt.clone(), create_graph=False).detach().numpy()

        err = np.abs(net_out - exact_vals)
        return {
            "operator":      operator,
            "max_error":     float(err.max()),
            "rms_error":     float(np.sqrt(np.mean(err**2))),
            "exact_range":   (float(exact_vals.min()), float(exact_vals.max())),
            "net_range":     (float(net_out.min()), float(net_out.max())),
        }

    # ── LaTeX report ───────────────────────────────────────────────────────

    def latex_report(self, field: SymbolicField) -> str:
        """Generate a LaTeX report of the field and all its operators."""
        lines = [
            r"\section{MNN Symbolic Field Report}",
            f"Field: ${field.name}$",
            "",
            r"\subsection{Field Expression}",
            f"${field.name}({', '.join(field.var_names)}) = {field.to_latex()}$",
            "",
        ]
        if field.is_scalar:
            g   = field.gradient()
            lap = field.laplacian()
            lines += [
                r"\subsection{Gradient}",
                f"$\\nabla {field.name} = {g.to_latex()}$",
                "",
                r"\subsection{Laplacian}",
                f"$\\nabla^2 {field.name} = {lap.to_latex()}$",
            ]
        else:
            div = field.divergence()
            lines += [
                r"\subsection{Divergence}",
                f"$\\nabla \\cdot {field.name} = {div.to_latex()}$",
            ]
            if len(field.components) == 3:
                curl = field.curl()
                lines += [
                    "",
                    r"\subsection{Curl}",
                    f"$\\nabla \\times {field.name} = {curl.to_latex()}$",
                ]
        return "\n".join(lines)

    def __repr__(self):
        return f"SymbolicValidator(dim={self.dim}, vars={self.var_names})"
