"""mnn.quantum_pde.neural_operator — Neural Operator Architecture.

Part 3: Learn function space → function space mappings. Instead of
learning one PDE solution, learn the operator governing an entire
family of PDEs.

Part 4: Quantum-Inspired Evolution — unitary propagation |Ψ(t+Δt)⟩ = U|Ψ(t)⟩.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class NeuralPDEOperator(nn.Module):
    """Learn a PDE operator mapping: input function → output function.

    Architecture: lift → integral kernel layers → project.
    Operates on discretized functions (batch, n_grid, channels).
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 width: int = 32, n_layers: int = 4, n_grid: int = 64):
        super().__init__()
        self.width = width
        self.n_grid = n_grid
        self.lift = nn.Linear(in_channels + 1, width)  # +1 for coordinate
        self.layers = nn.ModuleList([
            IntegralKernelLayer(width, n_grid) for _ in range(n_layers)
        ])
        self.project = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, out_channels),
        )

    def forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None):
        """x: (batch, n_grid, in_channels), returns (batch, n_grid, out_channels)."""
        B, N, C = x.shape
        if grid is None:
            grid = torch.linspace(0, 1, N, device=x.device).unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)
        h = self.lift(torch.cat([x, grid], dim=-1))
        for layer in self.layers:
            h = layer(h)
        return self.project(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class IntegralKernelLayer(nn.Module):
    """Integral kernel layer: combines local (1D conv) and global (spectral) transforms."""
    def __init__(self, width: int, n_grid: int, n_modes: int = 16):
        super().__init__()
        self.n_modes = min(n_modes, n_grid // 2)
        self.local_conv = nn.Conv1d(width, width, 1)
        # Spectral weights (complex)
        self.spectral_r = nn.Parameter(torch.randn(self.n_modes, width, width) * 0.02)
        self.spectral_i = nn.Parameter(torch.randn(self.n_modes, width, width) * 0.02)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_grid, width)"""
        residual = x
        # Local path
        local = self.local_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        # Spectral path
        x_ft = torch.fft.rfft(x, dim=1)
        n_modes = min(self.n_modes, x_ft.shape[1])
        x_ft_trunc = x_ft[:, :n_modes, :]
        # Complex multiply with spectral weights
        W = torch.complex(self.spectral_r[:n_modes], self.spectral_i[:n_modes])
        out_ft = torch.einsum("bmw,mwv->bmv", x_ft_trunc, W)
        # Pad back
        out_full = torch.zeros_like(x_ft)
        out_full[:, :n_modes, :] = out_ft
        spectral = torch.fft.irfft(out_full, n=x.shape[1], dim=1)
        # Combine
        return self.norm(F.gelu(local + spectral) + residual)


class QuantumEvolutionLayer(nn.Module):
    """Part 4: Quantum-inspired unitary evolution for PDE states.

    |Ψ(t+Δt)⟩ = U(Δt)|Ψ(t)⟩ where U is parameterized as exp(-iHΔt).
    Ensures norm preservation and reversibility.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        # Parameterize Hermitian H via skew-symmetric A: H = (A - A^T)/2
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.dt = nn.Parameter(torch.tensor(0.1))

    def _hamiltonian(self) -> torch.Tensor:
        """Construct Hermitian matrix."""
        return (self.A - self.A.T) / 2

    def _unitary(self) -> torch.Tensor:
        """U = exp(-i H dt)"""
        H = self._hamiltonian()
        return torch.matrix_exp(-1j * H.to(torch.cfloat) * self.dt)

    def forward(self, psi_r: torch.Tensor, psi_i: torch.Tensor):
        """Evolve complex state (real, imag) unitarily."""
        psi = torch.complex(psi_r, psi_i)
        U = self._unitary()
        evolved = psi @ U.T
        return evolved.real, evolved.imag

    def unitarity_error(self) -> float:
        U = self._unitary()
        err = torch.norm(U.conj().T @ U - torch.eye(self.state_dim, dtype=torch.cfloat))
        return err.item()


class QuantumPDENet(nn.Module):
    """Combined Neural Operator + Quantum Evolution for PDE solving.

    Lifts input to complex representation, applies quantum evolution layers,
    then projects back to solution space.
    """
    def __init__(self, n_grid: int = 64, in_channels: int = 1,
                 out_channels: int = 1, width: int = 32,
                 n_operator_layers: int = 3, n_evolution_steps: int = 2):
        super().__init__()
        self.operator = NeuralPDEOperator(in_channels, width, width, n_operator_layers, n_grid)
        self.evolution_layers = nn.ModuleList([
            QuantumEvolutionLayer(width) for _ in range(n_evolution_steps)
        ])
        self.project = nn.Linear(width, out_channels)

    def forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None):
        """x: (batch, n_grid, in_channels) → (batch, n_grid, out_channels)"""
        h = self.operator(x, grid)  # (batch, n_grid, width)
        r, i = h, torch.zeros_like(h)
        for evo in self.evolution_layers:
            r, i = evo(r, i)
        return self.project(r)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
