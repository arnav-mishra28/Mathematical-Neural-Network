"""mnn.quantum.geometric — Quantum geometric learning.

Part 5: Embeddings on curved quantum state manifolds.
Instead of flat vectors, states live on the Fubini-Study manifold
with curvature-aware distances and geodesics.
"""
from __future__ import annotations
import torch, torch.nn as nn, numpy as np
from typing import Optional, Dict, List, Tuple


class FubiniStudyMetric:
    """Fubini-Study metric on complex projective space CP^(n-1).

    The natural geometry of pure quantum states. Two states are
    close if their overlap is large: ds^2 = 1 - |<psi|phi>|^2.
    """
    @staticmethod
    def distance(psi: np.ndarray, phi: np.ndarray) -> float:
        """Fubini-Study distance: arccos(|<psi|phi>|)."""
        psi_n = psi / (np.linalg.norm(psi) + 1e-15)
        phi_n = phi / (np.linalg.norm(phi) + 1e-15)
        overlap = np.abs(np.vdot(psi_n, phi_n))
        return float(np.arccos(np.clip(overlap, 0, 1)))

    @staticmethod
    def distance_batch(psi: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Batch Fubini-Study distances."""
        psi_n = psi / (np.linalg.norm(psi, axis=-1, keepdims=True) + 1e-15)
        phi_n = phi / (np.linalg.norm(phi, axis=-1, keepdims=True) + 1e-15)
        overlaps = np.abs(np.sum(psi_n.conj() * phi_n, axis=-1))
        return np.arccos(np.clip(overlaps, 0, 1))

    @staticmethod
    def metric_tensor(psi: np.ndarray) -> np.ndarray:
        """Fubini-Study metric tensor g_ij at state psi.

        g_ij = <d_i psi | d_j psi> - <d_i psi|psi><psi|d_j psi>
        For computational basis derivatives.
        """
        n = len(psi)
        psi_n = psi / (np.linalg.norm(psi) + 1e-15)
        proj = np.eye(n, dtype=complex) - np.outer(psi_n, psi_n.conj())
        g = proj.conj().T @ proj
        return np.real(g)

    @staticmethod
    def berry_phase(states: List[np.ndarray]) -> float:
        """Berry phase around a closed loop of states."""
        phase = 1.0 + 0j
        for i in range(len(states)):
            j = (i + 1) % len(states)
            si = states[i] / (np.linalg.norm(states[i]) + 1e-15)
            sj = states[j] / (np.linalg.norm(states[j]) + 1e-15)
            phase *= np.vdot(si, sj)
        return float(-np.angle(phase))

    @staticmethod
    def quantum_fisher_information(psi: np.ndarray, dpsi: np.ndarray) -> float:
        """Quantum Fisher information F = 4(Var(G)) for parameter estimation.

        F = 4[<dpsi|dpsi> - |<dpsi|psi>|^2]
        """
        psi_n = psi / (np.linalg.norm(psi) + 1e-15)
        t1 = np.real(np.vdot(dpsi, dpsi))
        t2 = np.abs(np.vdot(dpsi, psi_n)) ** 2
        return float(4 * (t1 - t2))


class QuantumEmbedding(nn.Module):
    """Project real features into quantum state space (complex unit vectors).

    x in R^d  -->  |psi(x)> in CP^(n-1)
    Learnable embedding via parameterized rotation angles.
    """
    def __init__(self, input_dim: int, state_dim: int,
                 n_layers: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.angle_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, state_dim),
                nn.Tanh(),
            ) for _ in range(n_layers)
        ])
        self.phase_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, state_dim),
                nn.Tanh(),
            ) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor):
        """Return (real, imag) of normalized quantum state."""
        real = torch.ones(x.shape[0], self.state_dim, device=x.device)
        imag = torch.zeros(x.shape[0], self.state_dim, device=x.device)

        for angle_net, phase_net in zip(self.angle_nets, self.phase_nets):
            theta = angle_net(x) * np.pi
            phi = phase_net(x) * np.pi
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            cos_p = torch.cos(phi)
            sin_p = torch.sin(phi)
            new_real = real * cos_t - imag * sin_t * cos_p
            new_imag = real * sin_t * sin_p + imag * cos_t
            real, imag = new_real, new_imag

        # Normalize to unit vector
        norm = torch.sqrt(real**2 + imag**2 + 1e-8).sum(dim=-1, keepdim=True).sqrt()
        real = real / (norm + 1e-8)
        imag = imag / (norm + 1e-8)
        return real, imag


class QuantumGeometricLayer(nn.Module):
    """Curvature-aware layer on the quantum state manifold.

    Operates on complex states while respecting the Fubini-Study geometry:
    - Tangent vector projection (stays on manifold)
    - Exponential map (geodesic step)
    """
    def __init__(self, state_dim: int, width: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.tangent_net = nn.Sequential(
            nn.Linear(2 * state_dim, width),
            nn.GELU(),
            nn.Linear(width, 2 * state_dim),
        )
        self.step_size = nn.Parameter(torch.tensor(0.1))

    def _project_tangent(self, psi_r: torch.Tensor, psi_i: torch.Tensor,
                         v_r: torch.Tensor, v_i: torch.Tensor):
        """Project v onto tangent space of |psi>: v - <psi|v>|psi>."""
        # <psi|v> = sum(psi_r*v_r + psi_i*v_i) + i*sum(psi_r*v_i - psi_i*v_r)
        ip_real = (psi_r * v_r + psi_i * v_i).sum(dim=-1, keepdim=True)
        ip_imag = (psi_r * v_i - psi_i * v_r).sum(dim=-1, keepdim=True)
        proj_r = v_r - (ip_real * psi_r - ip_imag * psi_i)
        proj_i = v_i - (ip_real * psi_i + ip_imag * psi_r)
        return proj_r, proj_i

    def forward(self, psi_r: torch.Tensor, psi_i: torch.Tensor):
        combined = torch.cat([psi_r, psi_i], dim=-1)
        tangent = self.tangent_net(combined)
        v_r, v_i = tangent.chunk(2, dim=-1)
        v_r, v_i = self._project_tangent(psi_r, psi_i, v_r, v_i)

        # Exponential map (first-order geodesic step)
        new_r = psi_r + self.step_size * v_r
        new_i = psi_i + self.step_size * v_i

        # Re-normalize (project back to manifold)
        norm = torch.sqrt((new_r**2 + new_i**2).sum(dim=-1, keepdim=True) + 1e-8)
        return new_r / norm, new_i / norm


class QuantumGeometricNetwork(nn.Module):
    """Full quantum geometric network: embed → geometric layers → decode.

    All intermediate representations live on the complex projective space
    CP^(n-1), with curvature-aware updates via tangent-space neural maps.
    """
    def __init__(self, input_dim: int, output_dim: int,
                 state_dim: int = 32, n_geo_layers: int = 4,
                 geo_width: int = 64):
        super().__init__()
        self.embedding = QuantumEmbedding(input_dim, state_dim)
        self.geo_layers = nn.ModuleList([
            QuantumGeometricLayer(state_dim, geo_width)
            for _ in range(n_geo_layers)
        ])
        self.decoder = nn.Linear(2 * state_dim, output_dim)

    def forward(self, x: torch.Tensor):
        r, i = self.embedding(x)
        for layer in self.geo_layers:
            r, i = layer(r, i)
        return self.decoder(torch.cat([r, i], dim=-1))

    def state_trajectory(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Return the quantum state at each layer (for analysis)."""
        states = []
        r, i = self.embedding(x)
        states.append((r.detach(), i.detach()))
        for layer in self.geo_layers:
            r, i = layer(r, i)
            states.append((r.detach(), i.detach()))
        return states

    def geodesic_distances(self, x: torch.Tensor) -> List[float]:
        """Compute Fubini-Study distance at each layer transition."""
        traj = self.state_trajectory(x)
        dists = []
        for k in range(len(traj) - 1):
            r1, i1 = traj[k]
            r2, i2 = traj[k + 1]
            psi1 = torch.complex(r1, i1)
            psi2 = torch.complex(r2, i2)
            overlap = torch.abs((psi1.conj() * psi2).sum(dim=-1)).mean()
            dist = torch.acos(torch.clamp(overlap, 0, 1))
            dists.append(float(dist.item()))
        return dists

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return self(torch.tensor(x, dtype=torch.float32)).numpy()
