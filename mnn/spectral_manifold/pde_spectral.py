"""mnn.spectral_manifold.pde_spectral — PDE Spectral Solvers & Theorem Topology.

Part 8: PDE Spectral Solvers — PDEs simplify in spectral space.
Heat equation becomes decoupled spectral evolution.

Part 9: Theorem Spectral Topology — apply spectral analysis to theorem
networks to detect clusters, proof bottlenecks, and hidden bridges.
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

from .laplacian import ManifoldLaplacian, SpectralDecomposition


class SpectralPDEEvolver:
    """Part 8: Solve PDEs efficiently in spectral space.

    Many PDEs decouple in the eigenbasis: each mode evolves independently.
    """
    def __init__(self, spectrum: SpectralDecomposition):
        self.spectrum = spectrum

    def solve_heat(self, u0: np.ndarray, alpha: float = 1.0,
                    dt: float = 0.01, n_steps: int = 100) -> np.ndarray:
        """Solve heat equation ∂u/∂t = α Δu in spectral space.

        Each mode decays: c_i(t) = c_i(0) exp(-α λ_i t).
        """
        coeffs = self.spectrum.project(u0)
        trajectory = [u0.copy()]
        for step in range(n_steps):
            t = (step + 1) * dt
            evolved = coeffs * np.exp(-alpha * self.spectrum.eigenvalues * t)
            trajectory.append(self.spectrum.reconstruct(evolved))
        return np.array(trajectory)

    def solve_wave(self, u0: np.ndarray, v0: np.ndarray,
                    c: float = 1.0, dt: float = 0.01,
                    n_steps: int = 100) -> np.ndarray:
        """Solve wave equation ∂²u/∂t² = c² Δu in spectral space.

        Each mode oscillates: c_i(t) = A_i cos(ω_i t) + B_i sin(ω_i t).
        """
        c_u = self.spectrum.project(u0)
        c_v = self.spectrum.project(v0)
        omega = c * self.spectrum.frequencies

        trajectory = [u0.copy()]
        for step in range(n_steps):
            t = (step + 1) * dt
            cos_wt = np.cos(omega * t)
            sin_wt = np.sin(omega * t)
            sinc_wt = np.where(omega > 1e-15, sin_wt / omega, t)
            evolved = c_u * cos_wt + c_v * sinc_wt
            trajectory.append(self.spectrum.reconstruct(evolved))
        return np.array(trajectory)

    def solve_diffusion_reaction(self, u0: np.ndarray,
                                   alpha: float = 1.0,
                                   reaction: Optional[Callable] = None,
                                   dt: float = 0.01,
                                   n_steps: int = 100) -> np.ndarray:
        """Solve diffusion-reaction: ∂u/∂t = αΔu + R(u).

        Split-step: spectral diffusion + pointwise reaction.
        """
        u = u0.copy()
        trajectory = [u.copy()]
        for _ in range(n_steps):
            # Diffusion step (spectral)
            coeffs = self.spectrum.project(u)
            coeffs *= np.exp(-alpha * self.spectrum.eigenvalues * dt)
            u = self.spectrum.reconstruct(coeffs)
            # Reaction step (pointwise)
            if reaction is not None:
                u = u + dt * reaction(u)
            trajectory.append(u.copy())
        return np.array(trajectory)

    def operator_compression(self, n_modes: int = 10) -> Dict:
        """Compress PDE operator to dominant spectral modes."""
        evals = self.spectrum.eigenvalues[:n_modes]
        evecs = self.spectrum.eigenvectors[:, :n_modes]
        total_energy = np.sum(self.spectrum.eigenvalues)
        captured = np.sum(evals)
        return {
            "n_modes": n_modes,
            "eigenvalues": evals,
            "energy_captured": float(captured / (total_energy + 1e-15)),
            "compression_ratio": float(n_modes / self.spectrum.laplacian_obj.n),
        }


class TheoremSpectralTopology:
    """Part 9: Spectral analysis of theorem networks.

    Detect clusters, proof bottlenecks, hidden conceptual bridges,
    and cross-domain connections without explicit supervision.
    """
    def __init__(self, adjacency: np.ndarray,
                 node_names: Optional[List[str]] = None):
        self.adj = adjacency
        self.n = adjacency.shape[0]
        self.names = node_names or [f"node_{i}" for i in range(self.n)]
        self.laplacian = ManifoldLaplacian.from_adjacency(adjacency)
        self.spectrum = SpectralDecomposition(self.laplacian, min(30, self.n - 2))

    def theorem_clusters(self, n_clusters: int = 3) -> Dict[int, List[str]]:
        """Identify theorem communities via spectral clustering."""
        coords = self.spectrum.eigenvectors[:, 1:n_clusters + 1]
        from scipy.cluster.vq import kmeans2
        try:
            _, labels = kmeans2(coords, n_clusters, minit='points')
        except Exception:
            labels = np.zeros(self.n, dtype=int)

        clusters = {}
        for c in range(n_clusters):
            members = np.where(labels == c)[0]
            clusters[c] = [self.names[i] for i in members]
        return clusters

    def proof_bottlenecks(self, top_k: int = 5) -> List[Dict]:
        """Find proof bottlenecks via Fiedler vector analysis."""
        if self.spectrum.eigenvectors.shape[1] < 2:
            return []
        fiedler = self.spectrum.eigenvectors[:, 1]
        # Nodes near zero crossing = bottlenecks
        scores = 1.0 / (np.abs(fiedler) + 1e-8)
        top_idx = np.argsort(-scores)[:top_k]
        return [{"node": self.names[i], "index": int(i),
                 "fiedler_value": float(fiedler[i]),
                 "bottleneck_score": float(scores[i])}
                for i in top_idx]

    def hidden_bridges(self, threshold: float = 0.1) -> List[Dict]:
        """Discover hidden conceptual bridges between clusters.

        Pairs of nodes in different clusters that are spectrally close.
        """
        coords = self.spectrum.eigenvectors[:, 1:6]
        from scipy.spatial.distance import cdist
        dists = cdist(coords, coords)

        clusters = self.theorem_clusters(3)
        node_to_cluster = {}
        for c, members in clusters.items():
            for m in members:
                node_to_cluster[m] = c

        bridges = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                ni, nj = self.names[i], self.names[j]
                ci = node_to_cluster.get(ni, -1)
                cj = node_to_cluster.get(nj, -1)
                if ci != cj and ci >= 0 and cj >= 0 and dists[i, j] < threshold:
                    bridges.append({
                        "node_a": ni, "node_b": nj,
                        "cluster_a": ci, "cluster_b": cj,
                        "spectral_distance": float(dists[i, j]),
                    })
        bridges.sort(key=lambda x: x["spectral_distance"])
        return bridges

    def spectral_importance(self) -> List[Dict]:
        """Rank nodes by spectral importance (participation in low modes)."""
        low_modes = self.spectrum.eigenvectors[:, 1:6]
        importance = np.sum(low_modes**2, axis=1)
        ranked = np.argsort(-importance)
        return [{"node": self.names[i], "importance": float(importance[i])}
                for i in ranked]

    def connectivity_report(self) -> Dict:
        """Full connectivity analysis of the theorem network."""
        return {
            "n_theorems": self.n,
            "algebraic_connectivity": float(
                self.spectrum.eigenvalues[self.spectrum.eigenvalues > 1e-10][0]
                if np.any(self.spectrum.eigenvalues > 1e-10) else 0),
            "spectral_gap": self.spectrum.spectral_gap(),
            "effective_dimension": self.spectrum.effective_dimension(),
            "n_connected_components": int(
                np.sum(self.spectrum.eigenvalues < 1e-10)),
            "clusters": self.theorem_clusters(3),
            "bottlenecks": self.proof_bottlenecks(3),
        }
