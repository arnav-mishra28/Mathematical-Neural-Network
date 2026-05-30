"""mnn.spectral_manifold.embeddings — Graph Spectral Learning & Embeddings.

Part 3: Graph spectral analysis — reveal hidden clusters, global structure,
theorem communities, and latent mathematical domains.

Part 4: Spectral Embeddings — embed structures into eigenfunction coordinates
preserving topology, connectivity, and geometry.

Part 5: Manifold Harmonics — interpret frequencies physically:
low = global structure, high = local details.
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

from .laplacian import ManifoldLaplacian, SpectralDecomposition


class GraphSpectralAnalyzer:
    """Part 3: Analyze graphs via spectral theory.

    Reveals clusters, communities, bottlenecks, and global structure.
    """
    def __init__(self, adjacency: np.ndarray):
        self.adj = adjacency
        self.n = adjacency.shape[0]
        self.laplacian = ManifoldLaplacian.from_adjacency(adjacency)
        self.spectrum = SpectralDecomposition(self.laplacian, min(30, self.n - 2))

    def spectral_clustering(self, n_clusters: int = 2) -> np.ndarray:
        """Cluster nodes using spectral coordinates."""
        coords = self.spectrum.eigenvectors[:, 1:n_clusters + 1]
        # Simple k-means-style clustering
        from scipy.cluster.vq import kmeans2
        try:
            _, labels = kmeans2(coords, n_clusters, minit='points')
        except Exception:
            labels = np.zeros(self.n, dtype=int)
        return labels

    def fiedler_vector(self) -> np.ndarray:
        """Second eigenvector — algebraic connectivity / graph bisection."""
        if self.spectrum.eigenvectors.shape[1] > 1:
            return self.spectrum.eigenvectors[:, 1]
        return np.zeros(self.n)

    def algebraic_connectivity(self) -> float:
        """Second smallest eigenvalue — connectivity strength."""
        nonzero = self.spectrum.eigenvalues[self.spectrum.eigenvalues > 1e-10]
        return float(nonzero[0]) if len(nonzero) > 0 else 0.0

    def cheeger_constant_estimate(self) -> float:
        """Estimate Cheeger constant from spectral gap: h ≥ λ_2/2."""
        return self.algebraic_connectivity() / 2

    def detect_communities(self, n_communities: int = 3) -> Dict:
        """Detect communities and characterize them."""
        labels = self.spectral_clustering(n_communities)
        communities = {}
        for c in range(n_communities):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            # Internal connectivity
            sub_adj = self.adj[np.ix_(members, members)]
            internal = sub_adj.sum() / (len(members)**2 + 1e-15)
            communities[c] = {
                "members": members.tolist(),
                "size": len(members),
                "internal_density": float(internal),
            }
        return communities

    def bottleneck_nodes(self, top_k: int = 5) -> List[int]:
        """Find bottleneck nodes (high betweenness via Fiedler vector)."""
        fv = self.fiedler_vector()
        # Nodes near zero crossing of Fiedler vector are bottlenecks
        scores = 1.0 / (np.abs(fv) + 1e-8)
        return np.argsort(-scores)[:top_k].tolist()

    def summary(self) -> str:
        return (f"GraphSpectralAnalyzer(n={self.n}, "
                f"connectivity={self.algebraic_connectivity():.4f}, "
                f"gap={self.spectrum.spectral_gap():.4f})")


class SpectralEmbedding:
    """Part 4: Embed structures into spectral coordinates.

    Uses Laplacian eigenvectors as coordinates — preserves topology,
    connectivity, and geometry far better than Euclidean embeddings.
    """
    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self.spectrum: Optional[SpectralDecomposition] = None

    def fit(self, adjacency: np.ndarray) -> "SpectralEmbedding":
        lap = ManifoldLaplacian.from_adjacency(adjacency)
        self.spectrum = SpectralDecomposition(lap, self.n_components + 1)
        return self

    def fit_points(self, points: np.ndarray, k: int = 10) -> "SpectralEmbedding":
        lap = ManifoldLaplacian(points, k)
        self.spectrum = SpectralDecomposition(lap, self.n_components + 1)
        return self

    def transform(self) -> np.ndarray:
        """Return spectral coordinates (skip trivial eigenvector)."""
        if self.spectrum is None:
            raise ValueError("Must call fit() first")
        evecs = self.spectrum.eigenvectors
        return evecs[:, 1:self.n_components + 1]

    def fit_transform(self, adjacency: np.ndarray) -> np.ndarray:
        return self.fit(adjacency).transform()

    def embedding_quality(self, original_distances: np.ndarray) -> float:
        """Measure how well spectral embedding preserves distances (Spearman corr)."""
        coords = self.transform()
        from scipy.spatial.distance import pdist
        from scipy.stats import spearmanr
        emb_dists = pdist(coords)
        orig_dists = pdist(original_distances) if original_distances.ndim == 2 else original_distances
        if len(emb_dists) != len(orig_dists):
            orig_dists = pdist(original_distances)
        corr, _ = spearmanr(emb_dists, orig_dists)
        return float(corr)


class ManifoldHarmonics:
    """Part 5: Interpret manifold frequencies physically.

    Low frequencies = global structure, broad geometry.
    High frequencies = local details, fine curvature.
    Separate large-scale mathematical structure from local noise.
    """
    def __init__(self, spectrum: SpectralDecomposition):
        self.spectrum = spectrum

    def decompose_signal(self, signal: np.ndarray) -> Dict:
        """Decompose signal into frequency bands."""
        coeffs = self.spectrum.project(signal)
        n = len(coeffs)
        low_cut = max(1, n // 4)
        mid_cut = max(low_cut + 1, n // 2)

        low_coeffs = coeffs.copy()
        low_coeffs[low_cut:] = 0
        mid_coeffs = coeffs.copy()
        mid_coeffs[:low_cut] = 0
        mid_coeffs[mid_cut:] = 0
        high_coeffs = coeffs.copy()
        high_coeffs[:mid_cut] = 0

        return {
            "coefficients": coeffs,
            "low_frequency": self.spectrum.reconstruct(low_coeffs),
            "mid_frequency": self.spectrum.reconstruct(mid_coeffs),
            "high_frequency": self.spectrum.reconstruct(high_coeffs),
            "energy_low": float(np.sum(low_coeffs**2)),
            "energy_mid": float(np.sum(mid_coeffs**2)),
            "energy_high": float(np.sum(high_coeffs**2)),
            "energy_total": float(np.sum(coeffs**2)),
        }

    def spectral_energy_distribution(self, signal: np.ndarray) -> np.ndarray:
        """Energy at each frequency: |c_i|^2."""
        coeffs = self.spectrum.project(signal)
        return coeffs**2

    def denoise(self, signal: np.ndarray, keep_ratio: float = 0.5) -> np.ndarray:
        """Denoise by keeping only low frequencies."""
        n_keep = max(1, int(len(self.spectrum.eigenvalues) * keep_ratio))
        return self.spectrum.low_pass(signal, n_keep)

    def multiscale_representation(self, signal: np.ndarray,
                                    scales: List[float] = None) -> List[np.ndarray]:
        """Represent signal at multiple diffusion scales."""
        if scales is None:
            scales = [0.1, 0.5, 1.0, 2.0, 5.0]
        return [self.spectrum.heat_diffusion(signal, t) for t in scales]

    def frequency_response(self) -> Dict:
        """Characterize the manifold's frequency structure."""
        evals = self.spectrum.eigenvalues
        freqs = self.spectrum.frequencies
        return {
            "eigenvalues": evals,
            "frequencies": freqs,
            "spectral_gap": self.spectrum.spectral_gap(),
            "effective_dimension": self.spectrum.effective_dimension(),
            "max_frequency": float(freqs[-1]) if len(freqs) > 0 else 0,
            "bandwidth": float(freqs[-1] - freqs[0]) if len(freqs) > 1 else 0,
        }
