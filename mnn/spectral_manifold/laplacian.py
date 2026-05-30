"""mnn.spectral_manifold.laplacian — Geometry as Frequency.

Part 1: Laplace-Beltrami operator on manifolds — captures curvature,
connectivity, diffusion, and geometry.

Part 2: Spectral Decomposition — compute eigenfunctions/eigenvalues.
ΔM φ_i = λ_i φ_i — manifold harmonics and intrinsic frequencies.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from typing import Dict, List, Optional, Tuple


class ManifoldLaplacian:
    """Laplace-Beltrami operator on a discrete manifold (point cloud or mesh).

    Captures intrinsic geometry: curvature, connectivity, diffusion.
    """
    def __init__(self, points: np.ndarray, k_neighbors: int = 10,
                 sigma: Optional[float] = None):
        """
        points: (n, d) point cloud
        k_neighbors: nearest neighbors for graph construction
        sigma: heat kernel bandwidth (auto if None)
        """
        self.points = np.asarray(points, dtype=float)
        self.n = len(points)
        self.k = min(k_neighbors, self.n - 1)
        self.sigma = sigma
        self._build()

    def _build(self):
        """Build adjacency and Laplacian from point cloud."""
        from scipy.spatial.distance import cdist
        D = cdist(self.points, self.points)

        # k-NN adjacency
        self.adjacency = np.zeros((self.n, self.n))
        for i in range(self.n):
            neighbors = np.argsort(D[i])[1:self.k + 1]
            for j in neighbors:
                self.adjacency[i, j] = 1
                self.adjacency[j, i] = 1

        # Heat kernel weights
        if self.sigma is None:
            nonzero = D[self.adjacency > 0]
            self.sigma = np.median(nonzero) if len(nonzero) > 0 else 1.0

        self.weights = np.where(
            self.adjacency > 0,
            np.exp(-D**2 / (2 * self.sigma**2)),
            0.0
        )

        # Graph Laplacian: L = D - W
        self.degree = np.diag(self.weights.sum(axis=1))
        self.laplacian = self.degree - self.weights

        # Normalized Laplacian: L_sym = D^{-1/2} L D^{-1/2}
        d_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(self.degree)) + 1e-15))
        self.laplacian_normalized = d_inv_sqrt @ self.laplacian @ d_inv_sqrt

    @classmethod
    def from_adjacency(cls, adjacency: np.ndarray, weighted: bool = False):
        """Build from precomputed adjacency matrix."""
        obj = cls.__new__(cls)
        obj.n = adjacency.shape[0]
        obj.adjacency = adjacency
        obj.weights = adjacency if weighted else (adjacency > 0).astype(float)
        obj.degree = np.diag(obj.weights.sum(axis=1))
        obj.laplacian = obj.degree - obj.weights
        d_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(obj.degree)) + 1e-15))
        obj.laplacian_normalized = d_inv_sqrt @ obj.laplacian @ d_inv_sqrt
        obj.points = None
        obj.k = 0
        obj.sigma = 1.0
        return obj

    @property
    def matrix(self) -> np.ndarray:
        return self.laplacian

    @property
    def normalized_matrix(self) -> np.ndarray:
        return self.laplacian_normalized


class SpectralDecomposition:
    """Part 2: Eigendecomposition of the manifold Laplacian.

    ΔM φ_i = λ_i φ_i — manifold harmonics.
    """
    def __init__(self, laplacian: ManifoldLaplacian, n_components: int = 20):
        self.laplacian_obj = laplacian
        self.n_components = min(n_components, laplacian.n - 2)
        self._decompose()

    def _decompose(self):
        """Compute eigenvalues and eigenvectors."""
        L = self.laplacian_obj.laplacian_normalized
        n_comp = min(self.n_components, L.shape[0] - 2)
        if n_comp < 1:
            self.eigenvalues = np.array([0.0])
            self.eigenvectors = np.ones((L.shape[0], 1)) / np.sqrt(L.shape[0])
            return
        try:
            L_sparse = csr_matrix(L)
            evals, evecs = eigsh(L_sparse, k=n_comp + 1, which='SM')
        except Exception:
            evals, evecs = np.linalg.eigh(L)
            evals = evals[:n_comp + 1]
            evecs = evecs[:, :n_comp + 1]

        idx = np.argsort(evals)
        self.eigenvalues = evals[idx]
        self.eigenvectors = evecs[:, idx]

    @property
    def frequencies(self) -> np.ndarray:
        """Geometric frequencies: sqrt(λ_i)."""
        return np.sqrt(np.maximum(self.eigenvalues, 0))

    @property
    def harmonics(self) -> np.ndarray:
        """Manifold harmonics (eigenvectors), shape (n_points, n_components)."""
        return self.eigenvectors

    def spectral_gap(self) -> float:
        """Gap between first two non-trivial eigenvalues — connectivity measure."""
        nonzero = self.eigenvalues[self.eigenvalues > 1e-10]
        if len(nonzero) < 2:
            return 0.0
        return float(nonzero[1] - nonzero[0])

    def effective_dimension(self, threshold: float = 0.95) -> int:
        """Number of components capturing threshold fraction of spectral energy."""
        total = np.sum(self.eigenvalues)
        if total < 1e-15:
            return 1
        cumsum = np.cumsum(self.eigenvalues) / total
        return int(np.searchsorted(cumsum, threshold) + 1)

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        """Reconstruct signal from spectral coefficients: f = Σ c_i φ_i."""
        return self.eigenvectors @ coefficients

    def project(self, signal: np.ndarray) -> np.ndarray:
        """Project signal onto spectral basis: c_i = <φ_i, f>."""
        return self.eigenvectors.T @ signal

    def filter(self, signal: np.ndarray,
               filter_fn: Optional[callable] = None) -> np.ndarray:
        """Apply spectral filter: transform in frequency domain then reconstruct."""
        coeffs = self.project(signal)
        if filter_fn is not None:
            coeffs = filter_fn(self.eigenvalues) * coeffs
        return self.reconstruct(coeffs)

    def low_pass(self, signal: np.ndarray, cutoff: int = 10) -> np.ndarray:
        """Keep only lowest frequencies."""
        coeffs = self.project(signal)
        coeffs[cutoff:] = 0
        return self.reconstruct(coeffs)

    def high_pass(self, signal: np.ndarray, cutoff: int = 10) -> np.ndarray:
        """Keep only highest frequencies."""
        coeffs = self.project(signal)
        coeffs[:cutoff] = 0
        return self.reconstruct(coeffs)

    def heat_diffusion(self, signal: np.ndarray, t: float = 1.0) -> np.ndarray:
        """Diffuse signal on manifold: exp(-λ_i t) in spectral domain."""
        coeffs = self.project(signal)
        diffused = np.exp(-self.eigenvalues * t) * coeffs
        return self.reconstruct(diffused)

    def summary(self) -> str:
        return (f"SpectralDecomposition(n={self.laplacian_obj.n}, "
                f"components={len(self.eigenvalues)}, "
                f"gap={self.spectral_gap():.4f}, "
                f"eff_dim={self.effective_dimension()})")
