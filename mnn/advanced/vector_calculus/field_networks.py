"""
mnn.advanced.vector_calculus.field_networks
=============================================
Neural network architectures for scalar, vector, and tensor fields.

Key insight: the network IS the field.
  - ScalarFieldNet:  f : Rⁿ → R
  - VectorFieldNet:  F : Rⁿ → Rⁿ  (or Rᵐ)
  - TensorFieldNet:  T : Rⁿ → R^(n×n)  (rank-2 tensor at each point)

All networks support automatic differentiation — the FieldOperators
module computes ∇, ∇·, ∇×, ∇² by differentiating through these networks.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple


# ── Shared building blocks ────────────────────────────────────────────────────

class SinActivation(nn.Module):
    """sin activation — excellent for learning periodic and smooth fields."""
    def forward(self, x): return torch.sin(x)


class FieldBlock(nn.Module):
    """
    Residual block optimised for field learning.
    Uses LayerNorm + skip connection — essential for stable derivative computation.
    """
    def __init__(self, width: int, activation: str = "tanh"):
        super().__init__()
        acts = {
            "tanh":    nn.Tanh(),
            "sin":     SinActivation(),
            "gelu":    nn.GELU(),
            "silu":    nn.SiLU(),
            "swish":   nn.SiLU(),
            "softplus":nn.Softplus(),
        }
        self.linear = nn.Linear(width, width)
        self.norm   = nn.LayerNorm(width)
        self.act    = acts.get(activation, nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.linear(x)) + x)


class BaseFieldNet(nn.Module):
    """Shared backbone: embed → [FieldBlock × depth] → project."""

    def __init__(self, input_dim: int, output_dim: int,
                 width: int = 128, depth: int = 5,
                 activation: str = "tanh"):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.width      = width
        self.depth      = depth

        self.embed  = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width))
        self.blocks = nn.ModuleList([FieldBlock(width, activation) for _ in range(depth)])
        self.head   = nn.Linear(width, output_dim)

        # Xavier init — critical for well-behaved gradients
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.embed(x))
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._backbone(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32)
            return self.forward(t).numpy()


# ── Scalar Field Network ──────────────────────────────────────────────────────

class ScalarFieldNet(BaseFieldNet):
    """
    Neural scalar field  f : Rⁿ → R.

    Learns a smooth scalar-valued function over n-dimensional space.
    Can be differentiated to yield:
      - ∇f  (gradient vector field)
      - ∇²f (Laplacian scalar field)
      - Hessian matrix

    Example use cases:
      - Temperature field T(x,y,z)
      - Electric potential φ(x,y,z)
      - Stream function ψ(x,y)
      - Solution to Laplace/Poisson equations
    """

    def __init__(self, space_dim: int = 3,
                 width: int = 128, depth: int = 5,
                 activation: str = "tanh"):
        super().__init__(space_dim, 1, width, depth, activation)
        self.space_dim = space_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, space_dim) → (N, 1)"""
        return self.head(self._backbone(x))

    def scalar_value(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: return (N,) instead of (N,1)."""
        return self.forward(x).squeeze(-1)

    def __repr__(self):
        return (f"ScalarFieldNet(space_dim={self.space_dim}, "
                f"width={self.width}, depth={self.depth}, "
                f"params={self.count_parameters():,})")


# ── Vector Field Network ──────────────────────────────────────────────────────

class VectorFieldNet(BaseFieldNet):
    """
    Neural vector field  F : Rⁿ → Rᵐ.

    Learns a vector-valued function. When n=m, this represents a true
    vector field on Rⁿ (e.g., velocity field, electric field).

    Can be differentiated to yield:
      - Jacobian J_ij = ∂Fᵢ/∂xⱼ
      - Divergence ∇·F = Σ ∂Fᵢ/∂xᵢ
      - Curl ∇×F (for n=3)
      - Vector Laplacian ∇²F

    Example use cases:
      - Velocity field u(x,y,z) in fluid dynamics
      - Electromagnetic field E(x,y,z)
      - Gradient flow field
      - Incompressible (div-free) fields
    """

    def __init__(self, space_dim: int = 3,
                 width: int = 128, depth: int = 5,
                 activation: str = "tanh",
                 field_dim: Optional[int] = None):
        self.space_dim = space_dim
        self.field_dim = field_dim or space_dim
        super().__init__(space_dim, self.field_dim, width, depth, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, space_dim) → (N, field_dim)"""
        return self.head(self._backbone(x))

    def component(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """Return the i-th component Fᵢ(x). Shape: (N,)"""
        return self.forward(x)[:, i]

    def __repr__(self):
        return (f"VectorFieldNet(space={self.space_dim}→field={self.field_dim}, "
                f"w={self.width}, d={self.depth}, "
                f"params={self.count_parameters():,})")


# ── Divergence-Free Vector Field via Stream Function ─────────────────────────

class DivergenceFreeFieldNet(nn.Module):
    """
    Exactly divergence-free vector field in 2D via stream function ψ:
      F = (∂ψ/∂y, -∂ψ/∂x)   →   ∇·F = ∂²ψ/∂x∂y - ∂²ψ/∂y∂x = 0  ✓

    In 3D: use vector potential A such that F = ∇×A:
      F = curl(A)   →   ∇·F = ∇·(∇×A) = 0  ✓ (always)

    This guarantees ∇·F = 0 BY CONSTRUCTION — no training needed for
    the divergence constraint. We train the potential, the field is derived.
    """

    def __init__(self, space_dim: int = 3,
                 width: int = 128, depth: int = 5,
                 activation: str = "tanh"):
        super().__init__()
        self.space_dim = space_dim
        # Potential: ψ: Rⁿ → R (2D) or A: R³ → R³ (3D)
        potential_out = 1 if space_dim == 2 else space_dim
        self.potential = BaseFieldNet(space_dim, potential_out, width, depth, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute F = curl(A) or F = rot(ψ) via autograd.
        Returns the divergence-free field at points x.
        """
        x = x.requires_grad_(True)
        if self.space_dim == 2:
            # F = (∂ψ/∂y, -∂ψ/∂x)
            psi = self.potential(x)             # (N,1)
            grad_psi = torch.autograd.grad(
                psi.sum(), x, create_graph=True, retain_graph=True
            )[0]                                # (N,2)
            Fx =  grad_psi[:, 1:2]             # ∂ψ/∂y
            Fy = -grad_psi[:, 0:1]             # -∂ψ/∂x
            return torch.cat([Fx, Fy], dim=-1) # (N,2)
        else:
            # F = ∇×A
            A = self.potential(x)              # (N,3)
            Fx, Fy, Fz = [], [], []
            for i in range(3):
                gi = torch.autograd.grad(
                    A[:, i].sum(), x, create_graph=True, retain_graph=True
                )[0]                           # (N,3)
                Fx.append(gi[:, 0:1])
                Fy.append(gi[:, 1:2])
                Fz.append(gi[:, 2:3])
            # curl(A) = (∂A₃/∂y - ∂A₂/∂z, ∂A₁/∂z - ∂A₃/∂x, ∂A₂/∂x - ∂A₁/∂y)
            curl_x = Fz[1] - Fy[2]
            curl_y = Fx[2] - Fz[0]
            curl_z = Fy[0] - Fx[1]
            return torch.cat([curl_x, curl_y, curl_z], dim=-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (f"DivergenceFreeFieldNet(dim={self.space_dim}, "
                f"params={self.count_parameters():,}) [exact ∇·F=0]")


# ── Curl-Free Field via Scalar Potential ─────────────────────────────────────

class CurlFreeFieldNet(nn.Module):
    """
    Exactly curl-free (irrotational) vector field via scalar potential φ:
      F = ∇φ   →   ∇×F = ∇×(∇φ) = 0  ✓ (always)

    Used for:
      - Conservative force fields (gravity, electrostatics)
      - Potential flow in fluids
      - Gradient descent flows
    """

    def __init__(self, space_dim: int = 3,
                 width: int = 128, depth: int = 5,
                 activation: str = "tanh"):
        super().__init__()
        self.space_dim = space_dim
        self.potential = BaseFieldNet(space_dim, 1, width, depth, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """F = ∇φ — gradient of the learned potential."""
        x   = x.requires_grad_(True)
        phi = self.potential(x)                # (N,1)
        grad_phi = torch.autograd.grad(
            phi.sum(), x, create_graph=True, retain_graph=True
        )[0]                                   # (N, space_dim)
        return grad_phi

    def potential_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.potential(x).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (f"CurlFreeFieldNet(dim={self.space_dim}, "
                f"params={self.count_parameters():,}) [exact ∇×F=0]")


# ── Tensor Field Network ──────────────────────────────────────────────────────

class TensorFieldNet(BaseFieldNet):
    """
    Neural rank-2 tensor field  T : Rⁿ → R^(n×n).

    At each point x ∈ Rⁿ, returns an n×n tensor T(x).
    Used for:
      - Stress/strain tensors in elasticity
      - Metric tensor fields in differential geometry
      - Electromagnetic field tensor F_μν
      - Second-order correlation tensors

    Optional symmetry enforcement:
      - symmetric=True:      T = (T + Tᵀ)/2
      - traceless=True:      T → T - (Tr T / n) I
      - positive_definite:   T = Uᵀ U  (Cholesky-like)
    """

    def __init__(self, space_dim: int = 3,
                 width: int = 128, depth: int = 5,
                 activation: str = "tanh",
                 symmetric: bool = False,
                 traceless: bool = False):
        n = space_dim
        super().__init__(space_dim, n * n, width, depth, activation)
        self.space_dim  = space_dim
        self.n          = n
        self.symmetric  = symmetric
        self.traceless  = traceless

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, n) → (N, n, n) tensor field."""
        raw = self.head(self._backbone(x))                  # (N, n²)
        T   = raw.view(-1, self.n, self.n)                  # (N, n, n)

        if self.symmetric:
            T = (T + T.transpose(-1, -2)) / 2              # symmetrize

        if self.traceless:
            tr  = torch.diagonal(T, dim1=-2, dim2=-1).sum(-1, keepdim=True)  # (N,1)
            eye = torch.eye(self.n, device=x.device).unsqueeze(0)
            T   = T - (tr / self.n).unsqueeze(-1) * eye    # remove trace

        return T

    def component(self, x: torch.Tensor, i: int, j: int) -> torch.Tensor:
        """Return T_ij(x). Shape: (N,)"""
        return self.forward(x)[:, i, j]

    def trace(self, x: torch.Tensor) -> torch.Tensor:
        """Tr(T(x)). Shape: (N,)"""
        T = self.forward(x)
        return torch.diagonal(T, dim1=-2, dim2=-1).sum(-1)

    def determinant(self, x: torch.Tensor) -> torch.Tensor:
        """det(T(x)). Shape: (N,)"""
        return torch.linalg.det(self.forward(x))

    def __repr__(self):
        return (f"TensorFieldNet(space={self.space_dim}, rank=2, "
                f"sym={self.symmetric}, traceless={self.traceless}, "
                f"params={self.count_parameters():,})")
