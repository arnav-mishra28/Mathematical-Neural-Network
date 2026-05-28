"""mnn.quantum_pde.spectral — Spectral PDE Solving.

Part 5: Transform PDEs into frequency domain where many become simpler.
û(k) = ∫ u(x) e^{-ikx} dx. Global structures emerge naturally.

Part 6: Quantum Geometric Regularization — constrain solutions to
evolve on stable geometric manifolds with norm preservation,
smooth evolution, and curvature minimization.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class SpectralPDESolver:
    """Solve PDEs using spectral (Fourier) methods.

    Transforms spatial domain to frequency domain, applies operators
    as multiplications, then transforms back.
    """
    def __init__(self, n_grid: int, L: float = 2 * np.pi):
        self.n_grid = n_grid
        self.L = L
        self.dx = L / n_grid
        self.x = np.linspace(0, L, n_grid, endpoint=False)
        self.k = np.fft.fftfreq(n_grid, d=self.dx) * 2 * np.pi

    def fft(self, u: np.ndarray) -> np.ndarray:
        return np.fft.fft(u)

    def ifft(self, u_hat: np.ndarray) -> np.ndarray:
        return np.fft.ifft(u_hat)

    def derivative(self, u: np.ndarray, order: int = 1) -> np.ndarray:
        """Spectral derivative: multiply by (ik)^order in Fourier space."""
        u_hat = self.fft(u)
        return np.real(self.ifft((1j * self.k)**order * u_hat))

    def laplacian(self, u: np.ndarray) -> np.ndarray:
        return self.derivative(u, 2)

    def solve_heat(self, u0: np.ndarray, alpha: float, dt: float,
                    n_steps: int) -> np.ndarray:
        """Solve heat equation ∂u/∂t = α∇²u spectrally (exact in Fourier)."""
        u_hat = self.fft(u0)
        trajectory = [u0.copy()]
        for _ in range(n_steps):
            u_hat *= np.exp(-alpha * self.k**2 * dt)
            trajectory.append(np.real(self.ifft(u_hat)))
        return np.array(trajectory)

    def solve_wave(self, u0: np.ndarray, v0: np.ndarray,
                    c: float, dt: float, n_steps: int) -> np.ndarray:
        """Solve wave equation ∂²u/∂t² = c²∇²u spectrally."""
        u_hat = self.fft(u0)
        v_hat = self.fft(v0)
        omega = c * np.abs(self.k)
        trajectory = [u0.copy()]
        for step in range(n_steps):
            t = (step + 1) * dt
            cos_wt = np.cos(omega * dt)
            sin_wt = np.sin(omega * dt)
            sinc_wt = np.where(omega > 1e-15, sin_wt / omega, dt)
            u_hat_new = u_hat * cos_wt + v_hat * sinc_wt
            v_hat_new = -u_hat * omega * sin_wt + v_hat * cos_wt
            u_hat, v_hat = u_hat_new, v_hat_new
            trajectory.append(np.real(self.ifft(u_hat)))
        return np.array(trajectory)

    def solve_schrodinger(self, psi0: np.ndarray, V: np.ndarray,
                           dt: float, n_steps: int) -> np.ndarray:
        """Solve Schrödinger equation i∂ψ/∂t = -½∇²ψ + Vψ (split-step)."""
        psi = psi0.astype(complex)
        kinetic = np.exp(-0.5j * self.k**2 * dt / 2)
        trajectory = [psi.copy()]
        for _ in range(n_steps):
            # Half-step kinetic
            psi_hat = self.fft(psi)
            psi = self.ifft(kinetic * psi_hat)
            # Full-step potential
            psi *= np.exp(-1j * V * dt)
            # Half-step kinetic
            psi_hat = self.fft(psi)
            psi = self.ifft(kinetic * psi_hat)
            trajectory.append(psi.copy())
        return np.array(trajectory)

    def power_spectrum(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectrum |û(k)|²."""
        u_hat = self.fft(u)
        power = np.abs(u_hat)**2
        return self.k, power


class SpectralLayer(nn.Module):
    """Neural spectral layer: learnable filtering in Fourier space."""
    def __init__(self, in_channels: int, out_channels: int,
                 n_modes: int = 16):
        super().__init__()
        self.n_modes = n_modes
        self.weights_r = nn.Parameter(torch.randn(n_modes, in_channels, out_channels) * 0.02)
        self.weights_i = nn.Parameter(torch.randn(n_modes, in_channels, out_channels) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_grid, in_channels) → (batch, n_grid, out_channels)"""
        x_ft = torch.fft.rfft(x, dim=1)
        n_modes = min(self.n_modes, x_ft.shape[1])
        W = torch.complex(self.weights_r[:n_modes], self.weights_i[:n_modes])
        out_ft = torch.zeros(x.shape[0], x_ft.shape[1], self.weights_r.shape[2],
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :n_modes] = torch.einsum("bnc,ncm->bnm", x_ft[:, :n_modes], W)
        return torch.fft.irfft(out_ft, n=x.shape[1], dim=1) + self.bias


class SpectralPDENet(nn.Module):
    """Neural PDE solver operating entirely in spectral space."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 width: int = 32, n_layers: int = 4, n_modes: int = 16):
        super().__init__()
        self.lift = nn.Linear(in_channels + 1, width)
        self.layers = nn.ModuleList([
            SpectralBlock(width, n_modes) for _ in range(n_layers)
        ])
        self.project = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, out_channels))

    def forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        if grid is None:
            grid = torch.linspace(0, 1, N, device=x.device).view(1, N, 1).expand(B, -1, -1)
        h = self.lift(torch.cat([x, grid], dim=-1))
        for layer in self.layers:
            h = layer(h)
        return self.project(h)


class SpectralBlock(nn.Module):
    def __init__(self, width: int, n_modes: int):
        super().__init__()
        self.spectral = SpectralLayer(width, width, n_modes)
        self.local = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.gelu(self.spectral(x) + self.local(x)) + x)


# ---- Part 6: Geometric Regularization ----

class GeometricPDERegularizer(nn.Module):
    """Constrain PDE solutions to evolve on stable geometric manifolds.

    Losses: norm preservation, smoothness, curvature minimization.
    """
    def __init__(self, norm_weight: float = 1.0, smooth_weight: float = 0.5,
                 curvature_weight: float = 0.1):
        super().__init__()
        self.norm_weight = norm_weight
        self.smooth_weight = smooth_weight
        self.curvature_weight = curvature_weight

    def norm_preservation_loss(self, u: torch.Tensor, u_ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        """||u||² should be preserved (≈ ||u_ref||² or ≈ 1)."""
        norms = torch.sum(u**2, dim=-1).mean(dim=-1)
        if u_ref is not None:
            ref_norms = torch.sum(u_ref**2, dim=-1).mean(dim=-1)
            return ((norms - ref_norms)**2).mean()
        return ((norms - 1.0)**2).mean()

    def smoothness_loss(self, u: torch.Tensor) -> torch.Tensor:
        """Penalize sharp gradients: ||∂u/∂x||²."""
        grad = u[:, 1:, :] - u[:, :-1, :]
        return (grad**2).mean()

    def curvature_loss(self, u: torch.Tensor) -> torch.Tensor:
        """Penalize high curvature: ||∂²u/∂x²||²."""
        if u.shape[1] < 3:
            return torch.tensor(0.0, device=u.device)
        d2u = u[:, 2:, :] - 2 * u[:, 1:-1, :] + u[:, :-2, :]
        return (d2u**2).mean()

    def forward(self, u: torch.Tensor, u_ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = self.norm_weight * self.norm_preservation_loss(u, u_ref)
        loss += self.smooth_weight * self.smoothness_loss(u)
        loss += self.curvature_weight * self.curvature_loss(u)
        return loss
