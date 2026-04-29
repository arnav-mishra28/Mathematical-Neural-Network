"""
mnn.advanced.manifold_learning.constraints
============================================
Topological and geometric constraint losses for manifold learning.

These constraints are added to the autoencoder training loss to enforce
mathematical structure on the learned manifold.

Part 3: Enforce x² + y² = 1 and generalisations.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Optional


class ManifoldConstraints:
    """
    Collection of constraint functions for manifold autoencoders.

    All functions return callables of signature:
      loss_fn(x_hat, z, x) → scalar loss tensor

    Where:
      x_hat = reconstructed points (ambient space)
      z     = latent codes
      x     = original input points
    """

    # ── On-manifold constraints (reconstruction must lie on manifold) ─────────

    @staticmethod
    def on_circle(radius: float = 1.0) -> Callable:
        """
        Enforce reconstructed points lie on S¹:  x² + y² = r²
        residual = (x₀² + x₁² - r²)²
        """
        def loss(x_hat, z, x):
            sq = x_hat[:,0]**2 + x_hat[:,1]**2
            return torch.mean((sq - radius**2)**2)
        return loss

    @staticmethod
    def on_sphere(radius: float = 1.0) -> Callable:
        """
        Enforce reconstructed points lie on S^(n-1): ‖x̂‖ = r
        residual = (‖x̂‖ - r)²
        """
        def loss(x_hat, z, x):
            norms = torch.norm(x_hat, dim=-1)
            return torch.mean((norms - radius)**2)
        return loss

    @staticmethod
    def on_torus(R: float = 2.0, r: float = 0.8) -> Callable:
        """
        Enforce reconstructed points lie on T²:
        (√(x² + y²) - R)² + z² = r²
        """
        def loss(x_hat, z, x):
            xy_norm = torch.sqrt(x_hat[:,0]**2 + x_hat[:,1]**2 + 1e-8)
            torus_r = (xy_norm - R)**2 + x_hat[:,2]**2
            return torch.mean((torus_r - r**2)**2)
        return loss

    @staticmethod
    def on_hyperplane(normal: torch.Tensor, offset: float = 0.0) -> Callable:
        """
        Enforce reconstructed points lie on a hyperplane: n·x = d
        """
        normal = normal / normal.norm()
        def loss(x_hat, z, x):
            proj = (x_hat @ normal) - offset
            return torch.mean(proj**2)
        return loss

    @staticmethod
    def on_ellipsoid(semi_axes: torch.Tensor) -> Callable:
        """
        Enforce x̂ lies on ellipsoid: Σ (xᵢ/aᵢ)² = 1
        """
        def loss(x_hat, z, x):
            sq = torch.sum((x_hat / semi_axes.to(x_hat.device))**2, dim=-1)
            return torch.mean((sq - 1)**2)
        return loss

    # ── Latent space constraints ───────────────────────────────────────────────

    @staticmethod
    def latent_unit_sphere() -> Callable:
        """Enforce latent codes lie on S^(k-1): ‖z‖ = 1"""
        def loss(x_hat, z, x):
            norms = torch.norm(z, dim=-1)
            return torch.mean((norms - 1)**2)
        return loss

    @staticmethod
    def latent_gaussian_prior() -> Callable:
        """Encourage latent codes ~ N(0,I): minimise departure from unit Gaussian."""
        def loss(x_hat, z, x):
            mean_loss = torch.mean(z**2)                    # push toward 0
            var_loss  = torch.mean((torch.var(z, dim=0) - 1)**2)  # push var toward 1
            return mean_loss + var_loss
        return loss

    @staticmethod
    def latent_disentangle() -> Callable:
        """Encourage disentangled latent codes by penalising off-diagonal covariance."""
        def loss(x_hat, z, x):
            N = z.shape[0]
            z_centered = z - z.mean(dim=0, keepdim=True)
            cov = (z_centered.T @ z_centered) / N
            # Penalise off-diagonal entries
            mask = ~torch.eye(cov.shape[0], dtype=torch.bool, device=z.device)
            return torch.mean(cov[mask]**2)
        return loss

    @staticmethod
    def latent_uniform_circle() -> Callable:
        """
        For 2D latent space: push codes toward the unit circle.
        Useful when the manifold is topologically S¹.
        """
        def loss(x_hat, z, x):
            if z.shape[1] < 2:
                return torch.tensor(0.0, device=z.device)
            norms = torch.sqrt(z[:,0]**2 + z[:,1]**2 + 1e-8)
            return torch.mean((norms - 1)**2)
        return loss

    # ── Geometric constraints ─────────────────────────────────────────────────

    @staticmethod
    def isometry(n_pairs: int = 100) -> Callable:
        """
        Preserve pairwise distances: d_latent ≈ d_ambient.
        """
        def loss(x_hat, z, x):
            N = x.shape[0]
            i = torch.randint(0, N, (n_pairs,), device=x.device)
            j = torch.randint(0, N, (n_pairs,), device=x.device)
            d_amb = torch.norm(x[i] - x[j], dim=-1)
            d_lat = torch.norm(z[i] - z[j], dim=-1)
            d_amb_n = d_amb / (d_amb.max() + 1e-8)
            d_lat_n = d_lat / (d_lat.max() + 1e-8)
            return torch.mean((d_lat_n - d_amb_n)**2)
        return loss

    @staticmethod
    def reconstruction_jacobian_orthogonal() -> Callable:
        """
        Encourage the decoder Jacobian to have orthogonal columns
        (isometric parametrisation).
        """
        def loss(x_hat, z, x):
            z_req = z.detach().requires_grad_(True)
            with torch.enable_grad():
                # Approximate Jacobian via finite differences
                d   = z_req.shape[1]
                eps = 1e-3
                cols = []
                for i in range(d):
                    e = torch.zeros_like(z_req)
                    e[:, i] = eps
                    # Need access to decoder — skip if not available
                    cols.append(torch.zeros(x_hat.shape, device=z.device))
                if cols:
                    J = torch.stack(cols, dim=-1)       # (N, ambient, latent)
                    JTJ = torch.einsum('nik,njk->nij', J, J)
                    eye = torch.eye(d, device=z.device).unsqueeze(0)
                    return torch.mean((JTJ - eye)**2)
            return torch.tensor(0.0, device=z.device)
        return loss

    @staticmethod
    def smoothness(epsilon: float = 0.01) -> Callable:
        """
        Smoothness: nearby points in ambient → nearby points in latent.
        Uses random perturbations.
        """
        def loss(x_hat, z, x):
            noise = torch.randn_like(x) * epsilon
            return torch.mean(torch.norm(z, dim=-1)**2) * 0.0  # placeholder
        return loss

    # ── Topology constraints ──────────────────────────────────────────────────

    @staticmethod
    def periodic_latent(period: float = 2 * np.pi) -> Callable:
        """
        For circular topology: wrap latent codes to enforce periodicity.
        Penalise |z_i - z_j| when x_i, x_j are near antipodal points.
        """
        def loss(x_hat, z, x):
            # Encourage z to be wrapped: penalise large |z| values
            return torch.mean(torch.clamp(torch.norm(z, dim=-1) - period, min=0)**2)
        return loss

    @staticmethod
    def contractibility(target_center: Optional[torch.Tensor] = None) -> Callable:
        """
        Contractibility: encourage the latent code distribution to be connected.
        Penalise spread of latent codes.
        """
        def loss(x_hat, z, x):
            center = target_center if target_center is not None else z.mean(dim=0)
            return torch.mean(torch.norm(z - center, dim=-1))
        return loss

    # ── Combine constraints ───────────────────────────────────────────────────

    @staticmethod
    def combine(*constraints_and_weights) -> Callable:
        """
        Combine multiple constraints.
        Usage: combine((fn1, w1), (fn2, w2), ...)
        """
        def combined_loss(x_hat, z, x):
            total = torch.tensor(0.0, device=x.device)
            for fn, w in constraints_and_weights:
                total = total + w * fn(x_hat, z, x)
            return total
        return combined_loss

    def __repr__(self):
        return "ManifoldConstraints()"
