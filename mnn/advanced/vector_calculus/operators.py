"""
mnn.advanced.vector_calculus.operators
========================================
Autograd-based differential operators for MNN field networks.

Every operator here uses torch.autograd.grad — meaning these operators
work on ANY differentiable PyTorch network, not just specific architectures.

Operators
---------
  Scalar field  f: Rⁿ → R
    gradient       ∇f      : Rⁿ → Rⁿ
    laplacian      ∇²f     : Rⁿ → R
    hessian        ∇∇f     : Rⁿ → R^(n×n)
    directional_deriv  D_v f

  Vector field  F: Rⁿ → Rⁿ
    divergence     ∇·F     : Rⁿ → R
    curl           ∇×F     : R³ → R³
    jacobian       J_F     : Rⁿ → R^(n×n)
    vector_laplacian ∇²F   : Rⁿ → Rⁿ
    strain_rate    (J + Jᵀ)/2

  Tensor field  T: Rⁿ → R^(n×n)
    divergence     ∇·T     : Rⁿ → Rⁿ  (contraction)
    covariant_div          
    frobenius_laplacian

  Compound
    laplace_beltrami (on manifold)
    biharmonic     ∇⁴f
    d_alembertian  □f = (1/c²)∂²f/∂t² - ∇²f
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Optional, Tuple


class FieldOperators:
    """
    Autograd-based differential operators.
    All methods are static — call FieldOperators.gradient(net, x), etc.
    """

    # ── Scalar field operators ────────────────────────────────────────────────

    @staticmethod
    def gradient(scalar_net: nn.Module,
                 x: torch.Tensor,
                 create_graph: bool = True) -> torch.Tensor:
        """
        ∇f(x) — gradient of scalar field.

        Parameters
        ----------
        scalar_net : network f: (N, n) → (N, 1)
        x          : (N, n) input points
        Returns    : (N, n) gradient vectors
        """
        x   = x.requires_grad_(True)
        f   = scalar_net(x)
        if f.shape[-1] != 1:
            raise ValueError(f"Expected scalar output, got shape {f.shape}")
        grad = torch.autograd.grad(
            f.sum(), x, create_graph=create_graph, retain_graph=True
        )[0]
        return grad    # (N, n)

    @staticmethod
    def laplacian(scalar_net: nn.Module,
                  x: torch.Tensor,
                  create_graph: bool = True) -> torch.Tensor:
        """
        ∇²f(x) = Σᵢ ∂²f/∂xᵢ² — Laplacian of scalar field.

        Returns : (N, 1)
        """
        x    = x.requires_grad_(True)
        f    = scalar_net(x)
        grad = torch.autograd.grad(
            f.sum(), x, create_graph=True, retain_graph=True
        )[0]                                # (N, n)
        lap  = torch.zeros(x.shape[0], 1, device=x.device)
        for i in range(x.shape[1]):
            gi   = grad[:, i]               # (N,)
            d2fi = torch.autograd.grad(
                gi.sum(), x, create_graph=create_graph, retain_graph=True
            )[0][:, i]                      # ∂²f/∂xᵢ²
            lap[:, 0] += d2fi
        return lap                          # (N, 1)

    @staticmethod
    def hessian(scalar_net: nn.Module,
                x: torch.Tensor) -> torch.Tensor:
        """
        H_ij = ∂²f/∂xᵢ∂xⱼ — Hessian matrix.

        Returns : (N, n, n)
        """
        n    = x.shape[1]
        x    = x.requires_grad_(True)
        f    = scalar_net(x)
        grad = torch.autograd.grad(
            f.sum(), x, create_graph=True, retain_graph=True
        )[0]                               # (N, n)
        H    = torch.zeros(x.shape[0], n, n, device=x.device)
        for i in range(n):
            gi = grad[:, i]
            H_row = torch.autograd.grad(
                gi.sum(), x, create_graph=True, retain_graph=True
            )[0]                           # (N, n)
            H[:, i, :] = H_row
        return H                           # (N, n, n)

    @staticmethod
    def directional_derivative(scalar_net: nn.Module,
                               x: torch.Tensor,
                               direction: torch.Tensor,
                               create_graph: bool = True) -> torch.Tensor:
        """
        D_v f(x) = ∇f(x) · v̂ — directional derivative.

        direction : (n,) or (N, n) unit vector
        Returns   : (N, 1)
        """
        grad = FieldOperators.gradient(scalar_net, x, create_graph)
        v    = direction / (direction.norm(dim=-1, keepdim=True) + 1e-12)
        if v.dim() == 1:
            v = v.unsqueeze(0).expand_as(grad)
        return (grad * v).sum(dim=-1, keepdim=True)

    @staticmethod
    def biharmonic(scalar_net: nn.Module,
                   x: torch.Tensor) -> torch.Tensor:
        """∇⁴f = ∇²(∇²f) — biharmonic (double Laplacian). Returns (N,1)."""
        # We build a proxy that computes the laplacian and then diff again
        class LapProxy(nn.Module):
            def __init__(self, net): super().__init__(); self.net=net
            def forward(self, x): return FieldOperators.laplacian(self.net, x, create_graph=True)
        proxy = LapProxy(scalar_net)
        return FieldOperators.laplacian(proxy, x.detach().requires_grad_(True), create_graph=False)

    @staticmethod
    def d_alembertian(scalar_net: nn.Module,
                      xt: torch.Tensor,
                      time_idx: int = -1,
                      c: float = 1.0) -> torch.Tensor:
        """
        □f = (1/c²)∂²f/∂t² - ∇²f  (wave operator).

        xt       : (N, n+1) where last column (or time_idx) is time
        time_idx : which column is time
        Returns  : (N, 1)
        """
        xt   = xt.requires_grad_(True)
        f    = scalar_net(xt)
        grad = torch.autograd.grad(
            f.sum(), xt, create_graph=True, retain_graph=True
        )[0]                                    # (N, n+1)
        # ∂²f/∂t²
        dt   = grad[:, time_idx]
        d2t  = torch.autograd.grad(
            dt.sum(), xt, create_graph=True, retain_graph=True
        )[0][:, time_idx:time_idx+1]            # (N,1)
        # ∇²f (spatial only)
        spatial_idx = [i for i in range(xt.shape[1]) if i != (time_idx % xt.shape[1])]
        lap = torch.zeros(xt.shape[0], 1, device=xt.device)
        for i in spatial_idx:
            gi  = grad[:, i]
            d2i = torch.autograd.grad(
                gi.sum(), xt, create_graph=True, retain_graph=True
            )[0][:, i]
            lap[:, 0] += d2i
        return (1/c**2) * d2t - lap

    # ── Vector field operators ────────────────────────────────────────────────

    @staticmethod
    def divergence(vector_net: nn.Module,
                   x: torch.Tensor,
                   create_graph: bool = True) -> torch.Tensor:
        """
        ∇·F(x) = Σᵢ ∂Fᵢ/∂xᵢ — divergence of vector field.

        vector_net : F: (N, n) → (N, n)
        Returns    : (N, 1)
        """
        x    = x.requires_grad_(True)
        F    = vector_net(x)                    # (N, n)
        n    = x.shape[1]
        div  = torch.zeros(x.shape[0], 1, device=x.device)
        for i in range(min(n, F.shape[1])):
            dFi = torch.autograd.grad(
                F[:, i].sum(), x, create_graph=create_graph, retain_graph=True
            )[0][:, i:i+1]                     # ∂Fᵢ/∂xᵢ
            div += dFi
        return div                             # (N, 1)

    @staticmethod
    def curl(vector_net: nn.Module,
             x: torch.Tensor,
             create_graph: bool = True) -> torch.Tensor:
        """
        ∇×F(x) — curl of 3D vector field.

        (∇×F)ₓ = ∂Fz/∂y − ∂Fy/∂z
        (∇×F)ᵧ = ∂Fx/∂z − ∂Fz/∂x
        (∇×F)_z = ∂Fy/∂x − ∂Fx/∂y

        Returns : (N, 3)
        """
        if x.shape[1] != 3:
            raise ValueError(f"Curl requires 3D space, got dim={x.shape[1]}")
        x  = x.requires_grad_(True)
        F  = vector_net(x)                    # (N, 3)

        def _dFi_dxj(i, j):
            return torch.autograd.grad(
                F[:, i].sum(), x, create_graph=create_graph, retain_graph=True
            )[0][:, j]

        curl_x = _dFi_dxj(2, 1) - _dFi_dxj(1, 2)   # ∂Fz/∂y - ∂Fy/∂z
        curl_y = _dFi_dxj(0, 2) - _dFi_dxj(2, 0)   # ∂Fx/∂z - ∂Fz/∂x
        curl_z = _dFi_dxj(1, 0) - _dFi_dxj(0, 1)   # ∂Fy/∂x - ∂Fx/∂y

        return torch.stack([curl_x, curl_y, curl_z], dim=-1)   # (N, 3)

    @staticmethod
    def curl_2d(vector_net: nn.Module,
                x: torch.Tensor,
                create_graph: bool = True) -> torch.Tensor:
        """
        2D curl: (∇×F)_z = ∂Fy/∂x − ∂Fx/∂y  (scalar vorticity).
        Returns : (N, 1)
        """
        if x.shape[1] != 2:
            raise ValueError("curl_2d requires 2D input")
        x  = x.requires_grad_(True)
        F  = vector_net(x)                     # (N, 2)
        dFy_dx = torch.autograd.grad(
            F[:, 1].sum(), x, create_graph=create_graph, retain_graph=True
        )[0][:, 0:1]
        dFx_dy = torch.autograd.grad(
            F[:, 0].sum(), x, create_graph=create_graph, retain_graph=True
        )[0][:, 1:2]
        return dFy_dx - dFx_dy                 # (N, 1)

    @staticmethod
    def jacobian(vector_net: nn.Module,
                 x: torch.Tensor,
                 create_graph: bool = True) -> torch.Tensor:
        """
        J_ij = ∂Fᵢ/∂xⱼ — Jacobian matrix.
        Returns : (N, m, n)  where F: Rⁿ → Rᵐ
        """
        x  = x.requires_grad_(True)
        F  = vector_net(x)                    # (N, m)
        N, m = F.shape; n = x.shape[1]
        J  = torch.zeros(N, m, n, device=x.device)
        for i in range(m):
            Ji = torch.autograd.grad(
                F[:, i].sum(), x, create_graph=create_graph, retain_graph=True
            )[0]                              # (N, n)
            J[:, i, :] = Ji
        return J                              # (N, m, n)

    @staticmethod
    def vector_laplacian(vector_net: nn.Module,
                         x: torch.Tensor) -> torch.Tensor:
        """
        ∇²F — vector Laplacian (Laplacian applied component-wise).
        Returns : (N, m)
        """
        x   = x.requires_grad_(True)
        F   = vector_net(x)                   # (N, m)
        N,m = F.shape; n = x.shape[1]
        lap = torch.zeros_like(F)
        for comp in range(m):
            grad_comp = torch.autograd.grad(
                F[:, comp].sum(), x, create_graph=True, retain_graph=True
            )[0]                              # (N, n)
            for i in range(n):
                d2 = torch.autograd.grad(
                    grad_comp[:, i].sum(), x, create_graph=True, retain_graph=True
                )[0][:, i]
                lap[:, comp] += d2
        return lap                            # (N, m)

    @staticmethod
    def strain_rate(vector_net: nn.Module,
                    x: torch.Tensor) -> torch.Tensor:
        """
        Symmetric strain rate tensor  S = (J + Jᵀ)/2.
        Used in fluid mechanics and solid mechanics.
        Returns : (N, n, n)
        """
        J = FieldOperators.jacobian(vector_net, x)    # (N, m, n)
        return (J + J.transpose(-1, -2)) / 2

    @staticmethod
    def vorticity_tensor(vector_net: nn.Module,
                         x: torch.Tensor) -> torch.Tensor:
        """
        Anti-symmetric vorticity tensor  Ω = (J − Jᵀ)/2.
        Returns : (N, n, n)
        """
        J = FieldOperators.jacobian(vector_net, x)
        return (J - J.transpose(-1, -2)) / 2

    # ── Tensor field operators ────────────────────────────────────────────────

    @staticmethod
    def tensor_divergence(tensor_net: nn.Module,
                          x: torch.Tensor,
                          create_graph: bool = True) -> torch.Tensor:
        """
        ∇·T — divergence of rank-2 tensor field.
        (∇·T)ᵢ = Σⱼ ∂T_ij/∂xⱼ
        Returns : (N, n)  (a vector at each point)
        """
        x  = x.requires_grad_(True)
        T  = tensor_net(x)                    # (N, n, n)
        N, n, _ = T.shape
        div_T = torch.zeros(N, n, device=x.device)
        for i in range(n):
            for j in range(n):
                dTij_dxj = torch.autograd.grad(
                    T[:, i, j].sum(), x, create_graph=create_graph, retain_graph=True
                )[0][:, j]                    # ∂T_ij/∂xⱼ
                div_T[:, i] += dTij_dxj
        return div_T                          # (N, n)

    @staticmethod
    def frobenius_laplacian(tensor_net: nn.Module,
                            x: torch.Tensor) -> torch.Tensor:
        """
        ∇²T_ij = Σₖ ∂²T_ij/∂xₖ²  (Laplacian applied entry-wise).
        Returns : (N, n, n)
        """
        x  = x.requires_grad_(True)
        T  = tensor_net(x)
        N, n, _ = T.shape
        lap = torch.zeros_like(T)
        for i in range(n):
            for j in range(n):
                g1 = torch.autograd.grad(
                    T[:, i, j].sum(), x, create_graph=True, retain_graph=True
                )[0]
                for k in range(n):
                    d2 = torch.autograd.grad(
                        g1[:, k].sum(), x, create_graph=True, retain_graph=True
                    )[0][:, k]
                    lap[:, i, j] += d2
        return lap

    # ── Compound / physics operators ─────────────────────────────────────────

    @staticmethod
    def helmholtz_decomposition_residuals(
            vector_net: nn.Module, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute divergence and curl of a vector field.
        By Helmholtz's theorem, any smooth F = ∇φ + ∇×A.
        Returns (divergence (N,1), curl (N,3)) for use as constraints.
        """
        div  = FieldOperators.divergence(vector_net, x)
        curl = FieldOperators.curl(vector_net, x) if x.shape[1] == 3 else FieldOperators.curl_2d(vector_net, x)
        return div, curl

    @staticmethod
    def navier_stokes_residual(velocity_net: nn.Module,
                                pressure_net: nn.Module,
                                xt: torch.Tensor,
                                nu: float = 0.01) -> torch.Tensor:
        """
        Incompressible Navier-Stokes momentum residual:
        ∂u/∂t + (u·∇)u + ∇p − ν∇²u = 0

        xt         : (N, 4) = (x, y, z, t)
        velocity_net: (N,4)→(N,3)
        pressure_net: (N,4)→(N,1)
        Returns    : (N, 3) momentum residual
        """
        xt  = xt.requires_grad_(True)
        u   = velocity_net(xt)                # (N, 3)
        p   = pressure_net(xt)                # (N, 1)
        N   = xt.shape[0]
        res = torch.zeros(N, 3, device=xt.device)

        for i in range(3):
            # ∂uᵢ/∂t
            gui = torch.autograd.grad(u[:,i].sum(), xt, create_graph=True, retain_graph=True)[0]
            dui_dt = gui[:, 3]                            # time derivative
            # (u·∇)uᵢ = Σⱼ uⱼ ∂uᵢ/∂xⱼ
            conv = sum(u[:,j] * gui[:,j] for j in range(3))
            # ∂p/∂xᵢ
            gp = torch.autograd.grad(p.sum(), xt, create_graph=True, retain_graph=True)[0]
            dp_dxi = gp[:, i]
            # ν ∇²uᵢ
            lap_ui = torch.zeros(N, device=xt.device)
            for j in range(3):
                d2 = torch.autograd.grad(gui[:,j].sum(), xt, create_graph=True, retain_graph=True)[0][:,j]
                lap_ui += d2
            res[:, i] = dui_dt + conv + dp_dxi - nu * lap_ui
        return res

    def __repr__(self): return "FieldOperators()"
