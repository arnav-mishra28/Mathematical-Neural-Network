"""
mnn.advanced.manifold_learning.analysis
=========================================
Manifold analysis tools for MNN.

Computes topological and geometric properties of point clouds and
learned manifold representations.

Features
--------
  - Intrinsic dimension estimation (correlation dimension, MLE)
  - Persistent homology (Vietoris-Rips, approximate Betti numbers)
  - ISOMAP dimensionality reduction
  - Curvature profile analysis
  - Reconstruction quality metrics
  - Topology change detection
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from typing import Dict, List, Optional, Tuple


class ManifoldAnalyzer:
    """
    Analysis toolkit for manifold learning results.

    Takes a point cloud (or encoder/decoder pair) and computes
    topological and geometric summaries.
    """

    def __init__(self, points: Optional[np.ndarray] = None):
        self.points = points

    # ── Intrinsic dimension estimation ────────────────────────────────────────

    @staticmethod
    def intrinsic_dim_correlation(points: np.ndarray,
                                   n_scales: int = 15) -> float:
        """
        Estimate intrinsic dimension via correlation dimension (Grassberger-Procaccia).
        d = lim_{r→0} log C(r) / log r
        """
        dists  = pdist(points)
        r_min  = np.percentile(dists, 1)
        r_max  = np.percentile(dists, 30)
        r_vals = np.logspace(np.log10(r_min), np.log10(r_max), n_scales)
        C_r    = np.array([np.mean(dists < r) for r in r_vals])
        valid  = C_r > 0
        if valid.sum() < 3:
            return float(points.shape[1])
        slope, _ = np.polyfit(np.log(r_vals[valid]), np.log(C_r[valid]), 1)
        return float(max(slope, 0))

    @staticmethod
    def intrinsic_dim_mle(points: np.ndarray, k: int = 10) -> float:
        """
        Maximum Likelihood Estimation of intrinsic dimension (Levina-Bickel 2005).
        d̂ = (1/N) Σᵢ [ (1/(k-1)) Σⱼ log(Rₖ(i)/Rⱼ(i)) ]⁻¹
        """
        N    = len(points)
        D    = squareform(pdist(points))
        dims = []
        for i in range(N):
            sorted_dists = np.sort(D[i])[1:k+1]   # k nearest (excl. self)
            if sorted_dists[-1] < 1e-12:
                continue
            log_ratios = np.log(sorted_dists[-1] / sorted_dists[:-1] + 1e-15)
            if len(log_ratios) > 0 and log_ratios.mean() > 0:
                dims.append(1.0 / log_ratios.mean())
        return float(np.median(dims)) if dims else float(points.shape[1])

    @staticmethod
    def intrinsic_dim_pca(points: np.ndarray, variance_threshold: float = 0.95) -> int:
        """
        PCA-based intrinsic dimension: number of components capturing 95% variance.
        """
        centered = points - points.mean(axis=0)
        cov      = centered.T @ centered / len(points)
        evals    = np.linalg.eigvalsh(cov)[::-1]
        evals    = np.maximum(evals, 0)
        cumvar   = np.cumsum(evals) / (evals.sum() + 1e-15)
        return int(np.searchsorted(cumvar, variance_threshold) + 1)

    # ── Topology (Betti numbers via Vietoris-Rips) ────────────────────────────

    @staticmethod
    def betti_numbers_vietoris_rips(points: np.ndarray,
                                     epsilon: float = None,
                                     max_dim: int = 2) -> Dict[str, int]:
        """
        Approximate Betti numbers β₀, β₁ via Vietoris-Rips complex.
        β₀ = connected components, β₁ = independent loops.
        """
        import networkx as nx
        D = squareform(pdist(points))
        if epsilon is None:
            epsilon = np.percentile(D[D > 0], 10)

        G = nx.Graph()
        G.add_nodes_from(range(len(points)))
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if D[i,j] < epsilon:
                    G.add_edge(i, j)

        b0 = nx.number_connected_components(G)
        b1 = max(0, G.number_of_edges() - G.number_of_nodes() + b0)
        return {"beta_0": b0, "beta_1": b1, "epsilon": float(epsilon)}

    @staticmethod
    def persistence_diagram_0(points: np.ndarray,
                               n_scales: int = 20) -> List[Tuple[float, float]]:
        """
        0-dimensional persistence diagram: birth/death of connected components
        as epsilon increases. Returns list of (birth, death) pairs.
        """
        D      = squareform(pdist(points))
        max_d  = D.max()
        eps_vals = np.linspace(0, max_d, n_scales)
        import networkx as nx

        prev_b0 = len(points)
        pairs   = []
        for eps in eps_vals:
            G = nx.Graph()
            G.add_nodes_from(range(len(points)))
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    if D[i,j] < eps:
                        G.add_edge(i,j)
            curr_b0 = nx.number_connected_components(G)
            for _ in range(prev_b0 - curr_b0):
                pairs.append((0.0, float(eps)))
            prev_b0 = curr_b0
        return pairs

    # ── ISOMAP ────────────────────────────────────────────────────────────────

    @staticmethod
    def isomap(points: np.ndarray, n_components: int = 2,
               k_neighbors: int = 10) -> np.ndarray:
        """
        ISOMAP: nonlinear dimensionality reduction via geodesic distances.

        Steps:
          1. Build kNN graph
          2. Compute all-pairs shortest path (geodesic distance approx.)
          3. Classical MDS on geodesic distance matrix

        Returns: (N, n_components) embedding.
        """
        N    = len(points)
        D    = squareform(pdist(points))

        # kNN graph
        rows, cols, vals = [], [], []
        for i in range(N):
            knn = np.argsort(D[i])[:k_neighbors+1]
            for j in knn:
                if i != j:
                    rows.append(i); cols.append(j); vals.append(D[i,j])
        sparse = csr_matrix((vals, (rows, cols)), shape=(N,N))
        geo    = shortest_path(sparse, method='D', directed=False)

        # Handle disconnected components
        finite = geo[np.isfinite(geo)]
        if len(finite) > 0:
            geo[~np.isfinite(geo)] = finite.max() * 2

        # Classical MDS
        D2   = geo**2
        H    = np.eye(N) - np.ones((N,N))/N
        B    = -0.5 * H @ D2 @ H
        # Symmetrize
        B    = (B + B.T) / 2
        evals, evecs = np.linalg.eigh(B)
        idx  = np.argsort(evals)[::-1][:n_components]
        evals_pos = np.maximum(evals[idx], 0)
        return evecs[:, idx] * np.sqrt(evals_pos)[None, :]

    # ── Reconstruction quality ────────────────────────────────────────────────

    @staticmethod
    def reconstruction_quality(original: np.ndarray,
                                reconstructed: np.ndarray) -> Dict[str, float]:
        """Compute multiple reconstruction quality metrics."""
        err   = original - reconstructed
        mse   = float(np.mean(err**2))
        rmse  = float(np.sqrt(mse))
        mae   = float(np.mean(np.abs(err)))
        # Relative error
        scale = float(np.mean(np.linalg.norm(original, axis=-1)))
        rel   = rmse / (scale + 1e-10)
        # R² score
        ss_res = np.sum(err**2)
        ss_tot = np.sum((original - original.mean(axis=0))**2)
        r2     = float(1 - ss_res / (ss_tot + 1e-15))
        return {"mse": mse, "rmse": rmse, "mae": mae,
                "relative_error": rel, "r2_score": r2}

    @staticmethod
    def latent_coverage(z: np.ndarray, n_bins: int = 20) -> float:
        """
        Measure how uniformly the latent codes cover the latent space.
        Returns the fraction of bins occupied (coverage score).
        """
        d = z.shape[1]
        if d > 4:
            return float(np.mean(np.std(z, axis=0)))
        # Bin the latent space
        mins = z.min(axis=0); maxs = z.max(axis=0)
        bins_per_dim = max(2, int(n_bins**(1/d)))
        occupied = set()
        for pt in z:
            bin_idx = tuple(
                int(np.clip((pt[i]-mins[i])/(maxs[i]-mins[i]+1e-10)*bins_per_dim, 0, bins_per_dim-1))
                for i in range(d)
            )
            occupied.add(bin_idx)
        total = bins_per_dim**d
        return len(occupied) / total

    @staticmethod
    def manifold_smoothness(encoder: nn.Module,
                             points: np.ndarray,
                             epsilon: float = 0.01,
                             n_pairs: int = 200) -> float:
        """
        Smoothness: ratio of latent-space distances to ambient-space distances.
        A smooth manifold embedding has a bounded Lipschitz constant.
        """
        N = len(points)
        idx_i = np.random.choice(N, n_pairs); idx_j = np.random.choice(N, n_pairs)
        x_i   = points[idx_i]; x_j = points[idx_j]
        d_amb = np.linalg.norm(x_i - x_j, axis=-1)

        encoder.eval()
        with torch.no_grad():
            zi = encoder(torch.tensor(x_i, dtype=torch.float32)).numpy()
            zj = encoder(torch.tensor(x_j, dtype=torch.float32)).numpy()
        d_lat = np.linalg.norm(zi - zj, axis=-1)

        # Lipschitz: max(d_lat / d_amb)
        valid = d_amb > epsilon
        if valid.sum() == 0: return 0.0
        ratios = d_lat[valid] / d_amb[valid]
        return float(ratios.mean())

    # ── Full analysis report ──────────────────────────────────────────────────

    @staticmethod
    def full_report(points: np.ndarray,
                    reconstructed: Optional[np.ndarray] = None,
                    latent: Optional[np.ndarray] = None,
                    true_intrinsic_dim: Optional[int] = None) -> Dict:
        """
        Generate a complete manifold analysis report.
        """
        report = {}
        # Intrinsic dimension
        report["intrinsic_dim_corr"]   = ManifoldAnalyzer.intrinsic_dim_correlation(points[:500])
        report["intrinsic_dim_mle"]    = ManifoldAnalyzer.intrinsic_dim_mle(points[:500])
        report["intrinsic_dim_pca"]    = ManifoldAnalyzer.intrinsic_dim_pca(points)
        if true_intrinsic_dim:
            report["true_intrinsic_dim"] = true_intrinsic_dim

        # Topology
        eps_betti = np.percentile(pdist(points[:200]), 15)
        betti = ManifoldAnalyzer.betti_numbers_vietoris_rips(points[:200], eps_betti)
        report.update(betti)

        # Reconstruction
        if reconstructed is not None:
            rq = ManifoldAnalyzer.reconstruction_quality(points, reconstructed)
            report.update(rq)

        # Latent coverage
        if latent is not None:
            report["latent_coverage"] = ManifoldAnalyzer.latent_coverage(latent)
            report["latent_std"] = float(np.std(latent))

        return report

    def __repr__(self): return "ManifoldAnalyzer()"
