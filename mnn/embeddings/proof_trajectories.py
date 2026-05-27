"""mnn.embeddings.proof_trajectories — Proof Trajectories in Theorem Space.

Part 5: Model proofs as continuous paths T1 -> T2 -> T3 through
the theorem embedding space. Enables navigation of mathematical
landscapes, proof neighborhood search, and analogy discovery.
"""
from __future__ import annotations
import torch, torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ProofStep:
    """A single step in a proof trajectory."""
    state_real: np.ndarray  # complex embedding real part
    state_imag: np.ndarray  # complex embedding imag part
    rule_name: str = ""
    description: str = ""
    step_index: int = 0

    @property
    def state(self) -> np.ndarray:
        return self.state_real + 1j * self.state_imag

    def distance_to(self, other: "ProofStep") -> float:
        """Fubini-Study distance between proof states."""
        s1 = self.state / (np.linalg.norm(self.state) + 1e-15)
        s2 = other.state / (np.linalg.norm(other.state) + 1e-15)
        overlap = np.abs(np.vdot(s1, s2))
        return float(np.arccos(np.clip(overlap, 0, 1)))


@dataclass
class ProofTrajectory:
    """A proof as a path through theorem embedding space."""
    steps: List[ProofStep] = field(default_factory=list)
    name: str = ""
    metadata: Dict = field(default_factory=dict)

    def add_step(self, real: np.ndarray, imag: np.ndarray,
                 rule: str = "", description: str = ""):
        step = ProofStep(real, imag, rule, description, len(self.steps))
        self.steps.append(step)

    @property
    def length(self) -> int:
        return len(self.steps)

    def total_distance(self) -> float:
        """Total path length in theorem space."""
        return sum(self.steps[i].distance_to(self.steps[i+1])
                   for i in range(len(self.steps) - 1))

    def step_distances(self) -> List[float]:
        """Distance of each step."""
        return [self.steps[i].distance_to(self.steps[i+1])
                for i in range(len(self.steps) - 1)]

    def smoothness(self) -> float:
        """Smoothness: std of step sizes (lower = smoother)."""
        dists = self.step_distances()
        if len(dists) < 2:
            return 0.0
        return float(np.std(dists))

    def curvature(self) -> List[float]:
        """Discrete curvature at each interior point."""
        curvatures = []
        for i in range(1, len(self.steps) - 1):
            s0 = self.steps[i-1].state
            s1 = self.steps[i].state
            s2 = self.steps[i+1].state
            v1 = s1 - s0
            v2 = s2 - s1
            cos_angle = np.real(np.vdot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-15)
            curvatures.append(float(np.arccos(np.clip(cos_angle, -1, 1))))
        return curvatures

    def to_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (n_steps, dim) arrays of real and imag parts."""
        reals = np.stack([s.state_real for s in self.steps])
        imags = np.stack([s.state_imag for s in self.steps])
        return reals, imags

    def __repr__(self):
        return (f"ProofTrajectory({self.name}, {self.length} steps, "
                f"dist={self.total_distance():.4f})")


class ProofPathPredictor(nn.Module):
    """Neural network that predicts the next theorem state given current state + rule.

    Models proof dynamics: T_{n+1} = f(T_n, rule)
    """
    def __init__(self, embed_dim: int, n_rules: int, hidden: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.rule_embed = nn.Embedding(n_rules, hidden)
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim + hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * embed_dim),
        )

    def forward(self, state_r: torch.Tensor, state_i: torch.Tensor,
                rule_id: torch.Tensor):
        rule_vec = self.rule_embed(rule_id)
        combined = torch.cat([state_r, state_i, rule_vec], dim=-1)
        out = self.net(combined)
        next_r, next_i = out.chunk(2, dim=-1)
        # Normalize
        norm = torch.sqrt((next_r**2 + next_i**2).sum(dim=-1, keepdim=True) + 1e-8)
        return next_r / norm, next_i / norm


class ProofNavigator:
    """Navigate theorem space: interpolation, neighborhood search, analogies."""

    @staticmethod
    def interpolate(start_r: np.ndarray, start_i: np.ndarray,
                    end_r: np.ndarray, end_i: np.ndarray,
                    n_points: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Geodesic interpolation between two theorem states on CP^n."""
        s = (start_r + 1j * start_i)
        e = (end_r + 1j * end_i)
        s /= np.linalg.norm(s) + 1e-15
        e /= np.linalg.norm(e) + 1e-15

        # Slerp on complex unit sphere
        dot = np.real(np.vdot(s, e))
        dot = np.clip(dot, -1, 1)
        omega = np.arccos(abs(dot))

        if omega < 1e-8:
            # States are very close, use linear interpolation
            ts = np.linspace(0, 1, n_points)
            reals = np.stack([np.real((1-t)*s + t*e) for t in ts])
            imags = np.stack([np.imag((1-t)*s + t*e) for t in ts])
        else:
            # Phase-align
            if dot < 0:
                e = -e
                omega = np.pi - omega
            ts = np.linspace(0, 1, n_points)
            reals, imags = [], []
            for t in ts:
                interp = (np.sin((1-t)*omega) * s + np.sin(t*omega) * e) / np.sin(omega)
                interp /= np.linalg.norm(interp) + 1e-15
                reals.append(np.real(interp))
                imags.append(np.imag(interp))
            reals = np.stack(reals)
            imags = np.stack(imags)
        return reals, imags

    @staticmethod
    def neighborhood(center_r: np.ndarray, center_i: np.ndarray,
                     db_r: np.ndarray, db_i: np.ndarray,
                     radius: float, names: Optional[List[str]] = None) -> List[Dict]:
        """Find all theorems within a given Fubini-Study radius."""
        center = (center_r + 1j * center_i)
        center /= np.linalg.norm(center) + 1e-15
        db = db_r + 1j * db_i
        norms = np.linalg.norm(db, axis=-1, keepdims=True) + 1e-15
        db_norm = db / norms

        overlaps = np.abs(db_norm.conj() @ center)
        dists = np.arccos(np.clip(overlaps, 0, 1))

        results = []
        for i, d in enumerate(dists):
            if d <= radius:
                name = names[i] if names else f"theorem_{i}"
                results.append({"name": name, "distance": float(d)})
        results.sort(key=lambda x: x["distance"])
        return results

    @staticmethod
    def analogy(a_r: np.ndarray, a_i: np.ndarray,
                b_r: np.ndarray, b_i: np.ndarray,
                c_r: np.ndarray, c_i: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Analogy: A is to B as C is to D. Compute D = C + (B - A)."""
        a = a_r + 1j * a_i
        b = b_r + 1j * b_i
        c = c_r + 1j * c_i
        d = c + (b - a)
        d /= np.linalg.norm(d) + 1e-15
        return np.real(d), np.imag(d)
