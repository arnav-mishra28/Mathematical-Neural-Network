"""mnn.quantum.unitary — Unitary transformations and information-preserving layers.

Part 3: U^dag U = I constraint for stable, reversible neural dynamics.
Unitary layers preserve norm (no vanishing/exploding gradients),
enable reversible computation, and mirror quantum evolution.
"""
from __future__ import annotations
import torch, torch.nn as nn, numpy as np
from typing import Dict, List, Optional, Tuple


class UnitaryLayer(nn.Module):
    """A unitary linear transformation U: C^n -> C^n with U^dag U = I.

    Parameterized via the matrix exponential of a skew-Hermitian matrix:
    U = exp(A - A^dag) where A is a learnable complex matrix.
    This guarantees unitarity by construction.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.A_real = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.A_imag = nn.Parameter(torch.randn(dim, dim) * 0.1)

    def _skew_hermitian(self) -> torch.Tensor:
        A = torch.complex(self.A_real, self.A_imag)
        return A - A.conj().T

    def unitary_matrix(self) -> torch.Tensor:
        return torch.matrix_exp(self._skew_hermitian())

    def forward(self, real: torch.Tensor, imag: torch.Tensor):
        U = self.unitary_matrix()
        z = torch.complex(real, imag)
        Uz = z @ U.T
        return Uz.real, Uz.imag

    def unitarity_error(self) -> float:
        U = self.unitary_matrix()
        err = U.conj().T @ U - torch.eye(self.dim, dtype=U.dtype)
        return float(torch.norm(err).item())


class UnitaryBlock(nn.Module):
    """Unitary residual block for information-preserving dynamics."""
    def __init__(self, dim: int, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([UnitaryLayer(dim) for _ in range(n_layers)])
        self.phase_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, real: torch.Tensor, imag: torch.Tensor):
        for layer in self.layers:
            real, imag = layer(real, imag)
        # Apply learnable phase shift
        cos_p = torch.cos(self.phase_bias)
        sin_p = torch.sin(self.phase_bias)
        r_new = real * cos_p - imag * sin_p
        i_new = real * sin_p + imag * cos_p
        return r_new, i_new


class UnitaryNetwork(nn.Module):
    """Network of stacked unitary blocks for norm-preserving transformations.

    Encodes real input → complex, applies unitary blocks, decodes → real.
    """
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int = 64, n_blocks: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.encoder_r = nn.Linear(input_dim, hidden_dim)
        self.encoder_i = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([UnitaryBlock(hidden_dim) for _ in range(n_blocks)])
        self.decoder = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        r = self.encoder_r(x)
        i = self.encoder_i(x)
        for block in self.blocks:
            r, i = block(r, i)
        return self.decoder(torch.cat([r, i], dim=-1))

    def total_unitarity_error(self) -> float:
        err = 0.0
        for block in self.blocks:
            for layer in block.layers:
                err += layer.unitarity_error()
        return err

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UnitaryConstraintLoss(nn.Module):
    """Soft penalty to encourage unitarity: L = ||U^dag U - I||^2."""
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, model: nn.Module) -> torch.Tensor:
        total = torch.tensor(0.0)
        for m in model.modules():
            if isinstance(m, UnitaryLayer):
                U = m.unitary_matrix()
                eye = torch.eye(m.dim, dtype=U.dtype, device=U.device)
                total = total + torch.norm(U.conj().T @ U - eye) ** 2
        return self.weight * total


class ParameterizedUnitary(nn.Module):
    """Parameterized unitary via Givens rotations + diagonal phases.

    More parameter-efficient than full matrix exponential.
    """
    def __init__(self, dim: int, n_rotations: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.n_rot = n_rotations or (dim * (dim - 1) // 2)
        self.angles = nn.Parameter(torch.randn(self.n_rot) * 0.1)
        self.phases = nn.Parameter(torch.zeros(dim))
        # Pre-compute index pairs for Givens rotations
        self._pairs = []
        for i in range(dim):
            for j in range(i + 1, dim):
                self._pairs.append((i, j))
                if len(self._pairs) >= self.n_rot:
                    break
            if len(self._pairs) >= self.n_rot:
                break

    def unitary_matrix(self) -> torch.Tensor:
        U = torch.eye(self.dim, dtype=torch.cfloat)
        for k, (i, j) in enumerate(self._pairs):
            G = torch.eye(self.dim, dtype=torch.cfloat)
            c = torch.cos(self.angles[k])
            s = torch.sin(self.angles[k])
            G[i, i] = c; G[i, j] = -s
            G[j, i] = s; G[j, j] = c
            U = U @ G
        D = torch.diag(torch.exp(1j * self.phases))
        return U @ D

    def forward(self, real: torch.Tensor, imag: torch.Tensor):
        U = self.unitary_matrix()
        z = torch.complex(real, imag)
        Uz = z @ U.T
        return Uz.real, Uz.imag


class UnitaryTrainer:
    """Trainer for unitary networks with optional unitarity constraint."""
    def __init__(self, model: UnitaryNetwork, lr: float = 1e-3,
                 unitarity_weight: float = 0.01):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.unitarity_loss = UnitaryConstraintLoss(unitarity_weight)
        self.history: Dict[str, List[float]] = {"loss": [], "unitarity": []}

    def train(self, x: np.ndarray, y: np.ndarray,
              n_epochs: int = 1000, batch_size: int = 256,
              verbose: bool = True, print_every: int = 200) -> Dict:
        X = torch.tensor(x, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.float32)
        data_loss_fn = nn.MSELoss()

        for ep in range(n_epochs):
            self.model.train()
            idx = torch.randperm(len(X))[:batch_size]
            pred = self.model(X[idx])
            data_loss = data_loss_fn(pred, Y[idx])
            u_loss = self.unitarity_loss(self.model)
            loss = data_loss + u_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.history["loss"].append(loss.item())
            self.history["unitarity"].append(u_loss.item())

            if verbose and (ep + 1) % print_every == 0:
                print(f"  [Unitary] Epoch {ep+1} loss={loss.item():.6f} "
                      f"unitarity={u_loss.item():.6f}")

        return self.history
