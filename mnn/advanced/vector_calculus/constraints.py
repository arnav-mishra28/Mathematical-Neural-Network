"""
mnn.advanced.vector_calculus.constraints
==========================================
Mathematical constraints for field learning in MNN.

Each constraint returns a RESIDUAL TENSOR whose MSE loss is minimized.
Zero residual ↔ constraint exactly satisfied.

Available constraints
---------------------
  Scalar field constraints:
    harmonic            ∇²f = 0          (Laplace equation)
    poisson             ∇²f = g(x)       (Poisson equation)
    eikonal             |∇f| = 1         (Eikonal equation)

  Vector field constraints:
    divergence_free     ∇·F = 0          (incompressible / solenoidal)
    curl_free           ∇×F = 0          (irrotational / conservative)
    divergence_equals   ∇·F = g(x)       (prescribed divergence)
    curl_equals         ∇×F = G(x)       (prescribed curl)

  Coupled PDE constraints:
    maxwell_gauss_e     ∇·E = ρ/ε₀
    maxwell_gauss_b     ∇·B = 0
    maxwell_faraday     ∇×E = -∂B/∂t
    maxwell_ampere      ∇×B = μ₀J + μ₀ε₀∂E/∂t
    stokes              ∇²u = ∇p, ∇·u = 0
    heat_equation       ∂f/∂t = α∇²f
    wave_equation       ∂²f/∂t² = c²∇²f

  Tensor field constraints:
    symmetric_tensor    T = Tᵀ
    traceless_tensor    Tr(T) = 0
    einstein_vacuum     G_μν = 0         (vacuum Einstein equations, linearised)
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Callable, Optional
from .operators import FieldOperators


class FieldConstraints:
    """
    Collection of mathematical constraints for field networks.
    All return residual tensors for use in MSE loss computation.
    """

    # ── Scalar field constraints ──────────────────────────────────────────────

    @staticmethod
    def harmonic(scalar_net: nn.Module) -> Callable:
        """∇²f = 0  (Laplace equation). Residual: (N,1)"""
        def residual(x: torch.Tensor) -> torch.Tensor:
            return FieldOperators.laplacian(scalar_net, x)
        return residual

    @staticmethod
    def poisson(scalar_net: nn.Module,
                source_fn: Callable) -> Callable:
        """∇²f = g(x)  (Poisson equation). Residual: (N,1)"""
        def residual(x: torch.Tensor) -> torch.Tensor:
            lap    = FieldOperators.laplacian(scalar_net, x)
            source = source_fn(x)
            if source.dim() == 1: source = source.unsqueeze(-1)
            return lap - source
        return residual

    @staticmethod
    def eikonal(scalar_net: nn.Module) -> Callable:
        """|∇f| = 1  (Eikonal equation). Residual: (N,1)"""
        def residual(x: torch.Tensor) -> torch.Tensor:
            grad = FieldOperators.gradient(scalar_net, x)
            norm = torch.norm(grad, dim=-1, keepdim=True)
            return norm - 1.0
        return residual

    @staticmethod
    def biharmonic_constraint(scalar_net: nn.Module) -> Callable:
        """∇⁴f = 0  (biharmonic). Residual: (N,1)"""
        def residual(x: torch.Tensor) -> torch.Tensor:
            return FieldOperators.biharmonic(scalar_net, x.detach().clone())
        return residual

    # ── Vector field constraints ──────────────────────────────────────────────

    @staticmethod
    def divergence_free(vector_net: nn.Module) -> Callable:
        """
        ∇·F = 0  (incompressible / solenoidal field).
        This is the core constraint of incompressible fluid dynamics.
        Residual: (N,1)
        """
        def residual(x: torch.Tensor) -> torch.Tensor:
            return FieldOperators.divergence(vector_net, x)
        return residual

    @staticmethod
    def curl_free(vector_net: nn.Module) -> Callable:
        """
        ∇×F = 0  (irrotational / conservative field).
        Residual: (N,3) for 3D, (N,1) for 2D
        """
        def residual(x: torch.Tensor) -> torch.Tensor:
            if x.shape[1] == 3:
                return FieldOperators.curl(vector_net, x)
            return FieldOperators.curl_2d(vector_net, x)
        return residual

    @staticmethod
    def divergence_equals(vector_net: nn.Module,
                          source_fn: Callable) -> Callable:
        """∇·F = g(x). Residual: (N,1)"""
        def residual(x: torch.Tensor) -> torch.Tensor:
            div = FieldOperators.divergence(vector_net, x)
            g   = source_fn(x)
            if g.dim() == 1: g = g.unsqueeze(-1)
            return div - g
        return residual

    @staticmethod
    def curl_equals(vector_net: nn.Module,
                    prescribed_fn: Callable) -> Callable:
        """∇×F = G(x). Residual: (N,3) or (N,1)"""
        def residual(x: torch.Tensor) -> torch.Tensor:
            if x.shape[1] == 3:
                curl = FieldOperators.curl(vector_net, x)
            else:
                curl = FieldOperators.curl_2d(vector_net, x)
            target = prescribed_fn(x)
            return curl - target
        return residual

    @staticmethod
    def constant_divergence(vector_net: nn.Module,
                             value: float = 0.0) -> Callable:
        """∇·F = c (constant). Residual: (N,1)"""
        def residual(x: torch.Tensor) -> torch.Tensor:
            return FieldOperators.divergence(vector_net, x) - value
        return residual

    # ── PDE constraints ───────────────────────────────────────────────────────

    @staticmethod
    def heat_equation(scalar_net: nn.Module,
                      time_idx: int = -1,
                      alpha: float = 1.0) -> Callable:
        """
        ∂f/∂t = α∇²f  (heat / diffusion equation).
        Input x has one time coordinate.
        Residual: (N,1)
        """
        def residual(xt: torch.Tensor) -> torch.Tensor:
            xt   = xt.requires_grad_(True)
            f    = scalar_net(xt)
            grad = torch.autograd.grad(
                f.sum(), xt, create_graph=True, retain_graph=True
            )[0]
            t_col = time_idx % xt.shape[1]
            df_dt = grad[:, t_col:t_col+1]
            # Spatial Laplacian
            n = xt.shape[1]
            lap = torch.zeros_like(f)
            for i in range(n):
                if i == t_col: continue
                d2 = torch.autograd.grad(
                    grad[:,i].sum(), xt, create_graph=True, retain_graph=True
                )[0][:, i:i+1]
                lap += d2
            return df_dt - alpha * lap
        return residual

    @staticmethod
    def wave_equation(scalar_net: nn.Module,
                      time_idx: int = -1,
                      c: float = 1.0) -> Callable:
        """
        ∂²f/∂t² = c²∇²f  (wave equation).
        Residual: (N,1)
        """
        def residual(xt: torch.Tensor) -> torch.Tensor:
            return FieldOperators.d_alembertian(scalar_net, xt, time_idx, c)
        return residual

    @staticmethod
    def incompressible_continuity(velocity_net: nn.Module) -> Callable:
        """
        ∇·u = 0  (continuity for incompressible flow).
        Alias for divergence_free — explicit physics name.
        """
        return FieldConstraints.divergence_free(velocity_net)

    @staticmethod
    def maxwell_gauss_e(E_net: nn.Module,
                         rho_fn: Optional[Callable] = None,
                         eps0: float = 1.0) -> Callable:
        """∇·E = ρ/ε₀. Residual: (N,1)"""
        def residual(x: torch.Tensor) -> torch.Tensor:
            div_E = FieldOperators.divergence(E_net, x)
            rho   = rho_fn(x).unsqueeze(-1) if rho_fn else torch.zeros_like(div_E)
            return div_E - rho / eps0
        return residual

    @staticmethod
    def maxwell_gauss_b(B_net: nn.Module) -> Callable:
        """∇·B = 0  (no magnetic monopoles). Residual: (N,1)"""
        return FieldConstraints.divergence_free(B_net)

    @staticmethod
    def stokes_momentum(velocity_net: nn.Module,
                         pressure_net: nn.Module,
                         mu: float = 1.0) -> Callable:
        """
        Stokes flow: μ∇²u = ∇p.
        Residual: (N, n)
        """
        def residual(x: torch.Tensor) -> torch.Tensor:
            lap_u = FieldOperators.vector_laplacian(velocity_net, x)   # (N,n)
            x_    = x.requires_grad_(True)
            p     = pressure_net(x_)
            grad_p = torch.autograd.grad(
                p.sum(), x_, create_graph=True
            )[0]                                                        # (N,n)
            return mu * lap_u - grad_p
        return residual

    # ── Tensor field constraints ──────────────────────────────────────────────

    @staticmethod
    def symmetric_stress(tensor_net: nn.Module) -> Callable:
        """T = Tᵀ  (symmetric stress tensor). Residual: (N, n, n)"""
        def residual(x: torch.Tensor) -> torch.Tensor:
            T = tensor_net(x)
            return T - T.transpose(-1, -2)
        return residual

    @staticmethod
    def traceless_deviatoric(tensor_net: nn.Module) -> Callable:
        """Tr(T) = 0  (traceless / deviatoric tensor). Residual: (N,1)"""
        def residual(x: torch.Tensor) -> torch.Tensor:
            T  = tensor_net(x)                      # (N, n, n)
            tr = torch.diagonal(T, dim1=-2, dim2=-1).sum(-1, keepdim=True)
            return tr
        return residual

    @staticmethod
    def einstein_vacuum_linearised(metric_net: nn.Module) -> Callable:
        """
        Linearised vacuum Einstein equations: □h_μν = 0
        (gravitational waves in flat-space approximation).
        Applies d'Alembertian to each component of h_μν.
        Residual: (N, n, n)
        """
        def residual(xt: torch.Tensor) -> torch.Tensor:
            xt = xt.requires_grad_(True)
            h  = metric_net(xt)                     # (N, n, n)
            N, n, _ = h.shape
            box_h = torch.zeros_like(h)
            for i in range(n):
                for j in range(n):
                    class _Proxy(nn.Module):
                        def __init__(self, net_, i_, j_):
                            super().__init__()
                            self.net=net_; self.i=i_; self.j=j_
                        def forward(self, x):
                            return self.net(x)[:, self.i, self.j:self.j+1]
                    proxy = _Proxy(metric_net, i, j)
                    # Use the last index as time (convention)
                    box_h[:, i, j] = FieldOperators.d_alembertian(
                        proxy, xt.detach().clone(), time_idx=-1
                    ).squeeze(-1)
            return box_h
        return residual

    def __repr__(self): return "FieldConstraints()"
