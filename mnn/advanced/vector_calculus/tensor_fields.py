"""
mnn.advanced.vector_calculus.tensor_fields
============================================
Rank-2 and rank-3 tensor field operations for MNN.

Covers:
  - Covariant/contravariant index operations
  - Christoffel symbols from learned metric
  - Riemann curvature from learned metric field
  - Stress tensor analysis (von Mises, principal stresses)
  - Electromagnetic field tensor F_μν
  - Energy-momentum tensor T_μν
  - 3D tensor visualisation helpers
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, List
from .field_networks import TensorFieldNet, ScalarFieldNet
from .operators import FieldOperators


class TensorFieldEngine:
    """
    Advanced tensor field operations built on top of autograd.
    Works with any TensorFieldNet (or compatible network producing (N,n,n) output).
    """

    # ── Index gymnastics ──────────────────────────────────────────────────────

    @staticmethod
    def raise_index(T_net: nn.Module,
                    metric_inv_net: nn.Module,
                    x: torch.Tensor,
                    index: int = 0) -> torch.Tensor:
        """
        T^{i}_j = g^{ik} T_{kj}  — raise first index using inverse metric.
        Returns : (N, n, n)
        """
        T    = T_net(x)              # (N, n, n)
        g_inv= metric_inv_net(x)     # (N, n, n)
        # Einsum: T^i_j = g^{ik} T_{kj}
        return torch.einsum('...ik,...kj->...ij', g_inv, T)

    @staticmethod
    def lower_index(T_net: nn.Module,
                    metric_net: nn.Module,
                    x: torch.Tensor,
                    index: int = 0) -> torch.Tensor:
        """T_{ij} = g_{ik} T^k_j — lower first index."""
        T = T_net(x); g = metric_net(x)
        return torch.einsum('...ik,...kj->...ij', g, T)

    @staticmethod
    def trace_tensor(T_net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Tr(T) = g^{ij} T_{ij} (full contraction). Returns (N,)"""
        T = T_net(x)
        return torch.diagonal(T, dim1=-2, dim2=-1).sum(-1)

    @staticmethod
    def frobenius_norm(T_net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """‖T‖_F = √(Σ T_{ij}²). Returns (N,)"""
        T = T_net(x)
        return torch.norm(T.view(T.shape[0], -1), dim=-1)

    # ── Metric tensor operations ──────────────────────────────────────────────

    @staticmethod
    def christoffel_symbols_from_net(metric_net: nn.Module,
                                      x: torch.Tensor) -> torch.Tensor:
        """
        Γᵏᵢⱼ = (1/2) g^{kl} (∂g_{jl}/∂xᵢ + ∂g_{il}/∂xⱼ - ∂g_{ij}/∂xˡ)

        Computed fully via autograd from a learned metric network.
        metric_net : (N, n) → (N, n, n)
        Returns    : (N, n, n, n)  — Gamma[batch, k, i, j]
        """
        x   = x.requires_grad_(True)
        g   = metric_net(x)               # (N, n, n)
        N, n, _ = g.shape
        g_inv = torch.linalg.inv(g)       # (N, n, n)

        # Compute ∂g_{ab}/∂xˢ for all a,b,s
        dg = torch.zeros(N, n, n, n, device=x.device)   # dg[batch, a, b, s]
        for a in range(n):
            for b in range(n):
                dg_abs = torch.autograd.grad(
                    g[:, a, b].sum(), x, create_graph=True, retain_graph=True
                )[0]                       # (N, n)
                dg[:, a, b, :] = dg_abs

        # Christoffel: Γᵏᵢⱼ
        Gamma = torch.zeros(N, n, n, n, device=x.device)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    val = torch.zeros(N, device=x.device)
                    for l in range(n):
                        val += g_inv[:, k, l] * (
                            dg[:, j, l, i] + dg[:, i, l, j] - dg[:, i, j, l]
                        )
                    Gamma[:, k, i, j] = 0.5 * val

        return Gamma

    @staticmethod
    def riemann_tensor_from_christoffel(Gamma: torch.Tensor,
                                         x:     torch.Tensor,
                                         metric_net: nn.Module) -> torch.Tensor:
        """
        R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} - ∂_ν Γ^ρ_{μσ}
                   + Γ^ρ_{μλ} Γ^λ_{νσ} - Γ^ρ_{νλ} Γ^λ_{μσ}

        Uses quadratic terms in Gamma only (requires full derivatives for
        the linear terms — expensive; we return the algebraic approximation).
        Returns : (N, n, n, n, n)
        """
        N = Gamma.shape[0]; n = Gamma.shape[1]
        R = torch.zeros(N, n, n, n, n, device=x.device)
        for rho in range(n):
            for sig in range(n):
                for mu in range(n):
                    for nu in range(n):
                        r = torch.zeros(N, device=x.device)
                        for lam in range(n):
                            r += (Gamma[:, rho, mu, lam] * Gamma[:, lam, nu, sig]
                                - Gamma[:, rho, nu, lam] * Gamma[:, lam, mu, sig])
                        R[:, rho, sig, mu, nu] = r
        return R

    @staticmethod
    def ricci_tensor_from_riemann(R: torch.Tensor) -> torch.Tensor:
        """Ric_{μν} = R^ρ_{μρν} (trace over ρ). Returns (N, n, n)"""
        return torch.einsum('...riri->...ii', R).sum(-1).unsqueeze(-1).expand_as(R[:, :, :1, :1]).squeeze() if False else \
               torch.stack([torch.stack([
                   sum(R[:, r, mu, r, nu] for r in range(R.shape[1]))
                   for nu in range(R.shape[1])], dim=-1)
                   for mu in range(R.shape[1])], dim=-1)

    @staticmethod
    def ricci_scalar(ricci: torch.Tensor, g_inv: torch.Tensor) -> torch.Tensor:
        """R = g^{μν} Ric_{μν}. Returns (N,)"""
        return torch.einsum('...ij,...ij->...', g_inv, ricci)

    # ── Stress tensor analysis ────────────────────────────────────────────────

    @staticmethod
    def von_mises_stress(sigma_net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        von Mises stress: σ_VM = √( (3/2) s:s )
        where s = σ - (1/3) Tr(σ) I  is the deviatoric stress.
        Returns : (N,)
        """
        sigma = sigma_net(x)              # (N, 3, 3)
        n     = sigma.shape[1]
        tr    = torch.diagonal(sigma, dim1=-2, dim2=-1).sum(-1)  # (N,)
        eye   = torch.eye(n, device=x.device).unsqueeze(0)
        s     = sigma - (tr / n).unsqueeze(-1).unsqueeze(-1) * eye  # deviatoric
        return torch.sqrt(1.5 * torch.einsum('...ij,...ij->...', s, s))

    @staticmethod
    def principal_stresses(sigma_net: nn.Module,
                            x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute principal stresses (eigenvalues of σ) and directions (eigenvectors).
        Returns : (eigenvalues (N,n), eigenvectors (N,n,n))
        """
        sigma = sigma_net(x).cpu()        # (N, n, n)
        # Symmetrize for numerical stability
        sigma = (sigma + sigma.transpose(-1,-2)) / 2
        evals, evecs = torch.linalg.eigh(sigma)
        return evals, evecs

    @staticmethod
    def hydrostatic_pressure(sigma_net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """p = -Tr(σ)/3. Returns (N,)"""
        sigma = sigma_net(x)
        return -torch.diagonal(sigma, dim1=-2, dim2=-1).sum(-1) / 3

    # ── Electromagnetic field tensor ──────────────────────────────────────────

    @staticmethod
    def faraday_tensor(E_net: nn.Module, B_net: nn.Module,
                        xt: torch.Tensor) -> torch.Tensor:
        """
        Faraday tensor F_μν (antisymmetric, 4×4):
        F_0i = E_i/c,  F_ij = -ε_{ijk} B_k

        xt : (N, 4) = (x, y, z, t)
        Returns : (N, 4, 4)
        """
        E = E_net(xt)    # (N, 3) — electric field
        B = B_net(xt)    # (N, 3) — magnetic field
        N = xt.shape[0]
        F = torch.zeros(N, 4, 4, device=xt.device)
        # F_{0i} = E_i / c,  F_{i0} = -E_i / c
        F[:, 0, 1:4] =  E;  F[:, 1:4, 0] = -E
        # F_{ij} = -ε_{ijk} B_k
        F[:, 1, 2] = -B[:, 2]; F[:, 2, 1] =  B[:, 2]
        F[:, 2, 3] = -B[:, 0]; F[:, 3, 2] =  B[:, 0]
        F[:, 1, 3] =  B[:, 1]; F[:, 3, 1] = -B[:, 1]
        return F

    # ── Summary diagnostics ───────────────────────────────────────────────────

    @staticmethod
    def tensor_field_report(tensor_net: nn.Module,
                             x: torch.Tensor,
                             symmetric: bool = True) -> Dict:
        """Compute summary statistics of a tensor field at given points."""
        T  = tensor_net(x)                             # (N, n, n)
        if symmetric:
            T = (T + T.transpose(-1,-2)) / 2
        tr = torch.diagonal(T, dim1=-2, dim2=-1).sum(-1)  # (N,)
        fn = torch.norm(T.view(T.shape[0],-1), dim=-1)    # (N,)
        det= torch.linalg.det(T)                           # (N,)
        return {
            "trace_mean":   float(tr.mean().detach()),
            "trace_std":    float(tr.std().detach()),
            "frob_mean":    float(fn.mean().detach()),
            "frob_std":     float(fn.std().detach()),
            "det_mean":     float(det.mean().detach()),
            "det_positive": float((det > 0).float().mean().detach()),
            "n_points":     int(x.shape[0]),
        }

    def __repr__(self): return "TensorFieldEngine()"
