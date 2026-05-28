"""mnn.quantum_pde.states — Quantum State PDE Representation.

Part 1: Represent PDE solutions as quantum states |Ψ(x,t)⟩ evolving
in Hilbert space, not merely scalar fields u(x,t).

Part 2: Operator Formulation — Lu = f as learnable quantum operators.
"""
from __future__ import annotations
import torch, torch.nn as nn
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


class QuantumPDEState:
    """PDE solution as a quantum state in discretized Hilbert space.

    |Ψ(x,t)⟩ = Σ_i c_i(t) |φ_i(x)⟩

    Amplitudes are complex-valued, enabling interference and spectral structure.
    """
    def __init__(self, amplitudes: np.ndarray, grid: np.ndarray,
                 time: float = 0.0, metadata: Optional[Dict] = None):
        """
        amplitudes: (n_grid,) or (n_grid, n_components) complex array
        grid: (n_grid, n_dim) spatial coordinates
        """
        self.amplitudes = np.asarray(amplitudes, dtype=complex)
        self.grid = np.asarray(grid, dtype=float)
        self.time = time
        self.metadata = metadata or {}
        self.n_grid = self.grid.shape[0]

    @classmethod
    def from_real_field(cls, u: np.ndarray, grid: np.ndarray,
                        time: float = 0.0) -> "QuantumPDEState":
        """Create from real-valued field u(x)."""
        return cls(u.astype(complex), grid, time)

    @classmethod
    def gaussian_packet(cls, grid: np.ndarray, center: float = 0.0,
                         width: float = 1.0, k0: float = 0.0,
                         time: float = 0.0) -> "QuantumPDEState":
        """Create Gaussian wave packet."""
        if grid.ndim == 1:
            x = grid
        else:
            x = grid[:, 0]
        psi = np.exp(-(x - center)**2 / (2 * width**2)) * np.exp(1j * k0 * x)
        psi /= np.sqrt(np.sum(np.abs(psi)**2) * (x[1] - x[0]) + 1e-15)
        return cls(psi, grid, time)

    def norm(self) -> float:
        dx = self._grid_spacing()
        return float(np.sqrt(np.sum(np.abs(self.amplitudes)**2) * dx))

    def normalize(self) -> "QuantumPDEState":
        n = self.norm()
        if n > 1e-15:
            return QuantumPDEState(self.amplitudes / n, self.grid, self.time)
        return self

    def inner_product(self, other: "QuantumPDEState") -> complex:
        dx = self._grid_spacing()
        return complex(np.sum(np.conj(self.amplitudes) * other.amplitudes) * dx)

    def expectation_value(self, operator_matrix: np.ndarray) -> complex:
        """⟨Ψ|O|Ψ⟩"""
        O_psi = operator_matrix @ self.amplitudes
        dx = self._grid_spacing()
        return complex(np.sum(np.conj(self.amplitudes) * O_psi) * dx)

    def probability_density(self) -> np.ndarray:
        return np.abs(self.amplitudes)**2

    def phase_field(self) -> np.ndarray:
        return np.angle(self.amplitudes)

    def energy_density(self) -> np.ndarray:
        """Gradient-based energy density."""
        dx = self._grid_spacing()
        grad = np.gradient(self.amplitudes, dx)
        return np.abs(grad)**2

    def total_energy(self) -> float:
        dx = self._grid_spacing()
        return float(np.sum(self.energy_density()) * dx)

    def _grid_spacing(self) -> float:
        if self.grid.ndim == 1:
            return float(self.grid[1] - self.grid[0]) if len(self.grid) > 1 else 1.0
        return float(self.grid[1, 0] - self.grid[0, 0]) if len(self.grid) > 1 else 1.0

    def evolve(self, propagator: np.ndarray, dt: float) -> "QuantumPDEState":
        """Evolve: |Ψ(t+dt)⟩ = U(dt)|Ψ(t)⟩"""
        new_amps = propagator @ self.amplitudes
        return QuantumPDEState(new_amps, self.grid, self.time + dt)

    def __repr__(self):
        return (f"QuantumPDEState(n={self.n_grid}, t={self.time:.4f}, "
                f"norm={self.norm():.6f})")


class PDEOperator:
    """Learnable PDE operator L in Lu = f.

    Represents differential operators as matrices acting on discretized states.
    """
    def __init__(self, n_grid: int, dx: float, name: str = "L"):
        self.n_grid = n_grid
        self.dx = dx
        self.name = name

    @classmethod
    def laplacian_1d(cls, n: int, dx: float) -> "PDEOperator":
        """∇² in 1D with periodic boundary."""
        op = cls(n, dx, "∇²")
        D2 = np.zeros((n, n))
        for i in range(n):
            D2[i, i] = -2.0
            D2[i, (i+1) % n] = 1.0
            D2[i, (i-1) % n] = 1.0
        op._matrix = D2 / dx**2
        return op

    @classmethod
    def first_derivative_1d(cls, n: int, dx: float) -> "PDEOperator":
        """∂/∂x in 1D with periodic boundary (central difference)."""
        op = cls(n, dx, "∂/∂x")
        D1 = np.zeros((n, n))
        for i in range(n):
            D1[i, (i+1) % n] = 1.0
            D1[i, (i-1) % n] = -1.0
        op._matrix = D1 / (2 * dx)
        return op

    @classmethod
    def identity(cls, n: int, dx: float) -> "PDEOperator":
        op = cls(n, dx, "I")
        op._matrix = np.eye(n)
        return op

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def hermitian(self) -> bool:
        return bool(np.allclose(self._matrix, self._matrix.conj().T, atol=1e-10))

    def eigendecomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.hermitian:
            evals, evecs = np.linalg.eigh(self._matrix)
        else:
            evals, evecs = np.linalg.eig(self._matrix)
        return evals, evecs

    def spectral_radius(self) -> float:
        evals = np.linalg.eigvals(self._matrix)
        return float(np.max(np.abs(evals)))

    def apply(self, state: QuantumPDEState) -> np.ndarray:
        return self._matrix @ state.amplitudes

    def exponential(self, dt: float) -> np.ndarray:
        """exp(-i L dt) for unitary evolution."""
        from scipy.linalg import expm
        return expm(-1j * self._matrix * dt)

    def propagator(self, dt: float) -> np.ndarray:
        """exp(L dt) for diffusive evolution."""
        from scipy.linalg import expm
        return expm(self._matrix * dt)

    def combine(self, other: "PDEOperator", alpha: float = 1.0,
                beta: float = 1.0) -> "PDEOperator":
        """alpha * self + beta * other"""
        result = PDEOperator(self.n_grid, self.dx, f"({alpha}*{self.name}+{beta}*{other.name})")
        result._matrix = alpha * self._matrix + beta * other._matrix
        return result

    def __repr__(self):
        return f"PDEOperator({self.name}, n={self.n_grid}, hermitian={self.hermitian})"
