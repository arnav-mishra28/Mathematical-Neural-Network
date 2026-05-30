"""Tests for Spectral Manifold Learning (mnn.spectral_manifold)."""
import numpy as np
import torch
import pytest


# ===== Parts 1-2: Laplacian + Spectral Decomposition =====

class TestManifoldLaplacian:
    def test_from_points(self):
        from mnn.spectral_manifold.laplacian import ManifoldLaplacian
        pts = np.random.randn(30, 3)
        lap = ManifoldLaplacian(pts, k_neighbors=5)
        assert lap.laplacian.shape == (30, 30)
        assert np.allclose(lap.laplacian, lap.laplacian.T, atol=1e-10)

    def test_from_adjacency(self):
        from mnn.spectral_manifold.laplacian import ManifoldLaplacian
        adj = np.zeros((10, 10))
        for i in range(9):
            adj[i, i+1] = 1; adj[i+1, i] = 1
        lap = ManifoldLaplacian.from_adjacency(adj)
        assert lap.laplacian.shape == (10, 10)
        assert np.allclose(np.sum(lap.laplacian, axis=1), 0, atol=1e-10)


class TestSpectralDecomposition:
    def test_eigenvalues(self):
        from mnn.spectral_manifold.laplacian import ManifoldLaplacian, SpectralDecomposition
        adj = np.zeros((10, 10))
        for i in range(9):
            adj[i, i+1] = 1; adj[i+1, i] = 1
        lap = ManifoldLaplacian.from_adjacency(adj)
        spec = SpectralDecomposition(lap, 8)
        assert spec.eigenvalues[0] < 1e-6  # trivial eigenvalue ≈ 0
        assert len(spec.eigenvalues) >= 2

    def test_frequencies(self):
        from mnn.spectral_manifold.laplacian import ManifoldLaplacian, SpectralDecomposition
        pts = np.random.randn(20, 2)
        lap = ManifoldLaplacian(pts, 5)
        spec = SpectralDecomposition(lap, 10)
        freqs = spec.frequencies
        assert np.all(freqs >= 0)

    def test_low_pass(self):
        from mnn.spectral_manifold.laplacian import ManifoldLaplacian, SpectralDecomposition
        adj = np.zeros((20, 20))
        for i in range(19):
            adj[i, i+1] = 1; adj[i+1, i] = 1
        lap = ManifoldLaplacian.from_adjacency(adj)
        spec = SpectralDecomposition(lap, 15)
        signal = np.random.randn(20)
        filtered = spec.low_pass(signal, 5)
        assert filtered.shape == (20,)

    def test_heat_diffusion(self):
        from mnn.spectral_manifold.laplacian import ManifoldLaplacian, SpectralDecomposition
        adj = np.zeros((20, 20))
        for i in range(19):
            adj[i, i+1] = 1; adj[i+1, i] = 1
        lap = ManifoldLaplacian.from_adjacency(adj)
        spec = SpectralDecomposition(lap, 10)
        signal = np.zeros(20); signal[10] = 1.0
        diffused = spec.heat_diffusion(signal, t=1.0)
        # Diffusion should spread: max decreases
        assert np.max(np.abs(diffused)) <= np.max(np.abs(signal)) + 0.01


# ===== Part 3: Graph Spectral Analysis =====

class TestGraphSpectral:
    def test_analyzer(self):
        from mnn.spectral_manifold.embeddings import GraphSpectralAnalyzer
        adj = np.zeros((20, 20))
        for i in range(19):
            adj[i, i+1] = 1; adj[i+1, i] = 1
        analyzer = GraphSpectralAnalyzer(adj)
        assert analyzer.algebraic_connectivity() > 0

    def test_fiedler(self):
        from mnn.spectral_manifold.embeddings import GraphSpectralAnalyzer
        adj = np.zeros((10, 10))
        for i in range(9):
            adj[i, i+1] = 1; adj[i+1, i] = 1
        analyzer = GraphSpectralAnalyzer(adj)
        fv = analyzer.fiedler_vector()
        assert fv.shape == (10,)


# ===== Part 4: Spectral Embedding =====

class TestSpectralEmbedding:
    def test_fit_transform(self):
        from mnn.spectral_manifold.embeddings import SpectralEmbedding
        adj = np.zeros((15, 15))
        for i in range(14):
            adj[i, i+1] = 1; adj[i+1, i] = 1
        se = SpectralEmbedding(n_components=4)
        coords = se.fit_transform(adj)
        assert coords.shape == (15, 4)


# ===== Part 5: Manifold Harmonics =====

class TestManifoldHarmonics:
    def test_decompose(self):
        from mnn.spectral_manifold.laplacian import ManifoldLaplacian, SpectralDecomposition
        from mnn.spectral_manifold.embeddings import ManifoldHarmonics
        adj = np.zeros((20, 20))
        for i in range(19):
            adj[i, i+1] = 1; adj[i+1, i] = 1
        lap = ManifoldLaplacian.from_adjacency(adj)
        spec = SpectralDecomposition(lap, 15)
        mh = ManifoldHarmonics(spec)
        signal = np.random.randn(20)
        result = mh.decompose_signal(signal)
        assert "low_frequency" in result
        assert "high_frequency" in result
        # Energy conservation
        total = result["energy_low"] + result["energy_mid"] + result["energy_high"]
        assert abs(total - result["energy_total"]) < 0.01

    def test_denoise(self):
        from mnn.spectral_manifold.laplacian import ManifoldLaplacian, SpectralDecomposition
        from mnn.spectral_manifold.embeddings import ManifoldHarmonics
        adj = np.zeros((20, 20))
        for i in range(19):
            adj[i, i+1] = 1; adj[i+1, i] = 1
        lap = ManifoldLaplacian.from_adjacency(adj)
        spec = SpectralDecomposition(lap, 15)
        mh = ManifoldHarmonics(spec)
        signal = np.random.randn(20)
        denoised = mh.denoise(signal, keep_ratio=0.3)
        assert denoised.shape == (20,)


# ===== Part 6: Spectral Attention =====

class TestSpectralAttention:
    def test_forward(self):
        from mnn.spectral_manifold.spectral_attention import SpectralAttention
        attn = SpectralAttention(32, n_harmonics=8, n_heads=4)
        x = torch.randn(4, 10, 32)
        out = attn(x)
        assert out.shape == (4, 10, 32)


# ===== Part 7: Quantum Spectral =====

class TestQuantumSpectral:
    def test_forward(self):
        from mnn.spectral_manifold.spectral_attention import QuantumSpectralLayer
        qsl = QuantumSpectralLayer(16, n_harmonics=8)
        r, i = torch.randn(4, 16), torch.randn(4, 16)
        out_r, out_i = qsl(r, i)
        assert out_r.shape == (4, 16)
        # Output should be normalized
        norms = torch.sqrt((out_r**2 + out_i**2).sum(dim=-1))
        assert torch.allclose(norms, torch.ones(4), atol=0.01)

    def test_spectral_energy(self):
        from mnn.spectral_manifold.spectral_attention import QuantumSpectralLayer
        qsl = QuantumSpectralLayer(16, n_harmonics=8)
        r, i = torch.randn(4, 16), torch.randn(4, 16)
        energy = qsl.spectral_energy(r, i)
        assert energy.shape == (4, 8)
        assert torch.all(energy >= 0)


# ===== Part 8: PDE Spectral =====

class TestPDESpectral:
    def test_heat(self):
        from mnn.spectral_manifold.laplacian import ManifoldLaplacian, SpectralDecomposition
        from mnn.spectral_manifold.pde_spectral import SpectralPDEEvolver
        adj = np.zeros((20, 20))
        for i in range(19):
            adj[i, i+1] = 1; adj[i+1, i] = 1
        lap = ManifoldLaplacian.from_adjacency(adj)
        spec = SpectralDecomposition(lap, 15)
        evolver = SpectralPDEEvolver(spec)
        u0 = np.zeros(20); u0[10] = 1.0
        traj = evolver.solve_heat(u0, alpha=0.5, dt=0.1, n_steps=10)
        assert traj.shape == (11, 20)
        # Heat should diffuse
        assert np.max(np.abs(traj[-1])) < np.max(np.abs(traj[0])) + 0.01


# ===== Part 9: Theorem Spectral Topology =====

class TestTheoremTopology:
    def test_clusters(self):
        from mnn.spectral_manifold.pde_spectral import TheoremSpectralTopology
        # Two clusters connected by one edge
        adj = np.zeros((10, 10))
        for i in range(4):
            adj[i, i+1] = 1; adj[i+1, i] = 1
        for i in range(5, 9):
            adj[i, i+1] = 1; adj[i+1, i] = 1
        adj[4, 5] = 1; adj[5, 4] = 1  # bridge
        names = [f"T{i}" for i in range(10)]
        tst = TheoremSpectralTopology(adj, names)
        clusters = tst.theorem_clusters(2)
        assert len(clusters) == 2

    def test_bottlenecks(self):
        from mnn.spectral_manifold.pde_spectral import TheoremSpectralTopology
        adj = np.zeros((10, 10))
        for i in range(9):
            adj[i, i+1] = 1; adj[i+1, i] = 1
        tst = TheoremSpectralTopology(adj)
        bottlenecks = tst.proof_bottlenecks(3)
        assert len(bottlenecks) == 3

    def test_connectivity_report(self):
        from mnn.spectral_manifold.pde_spectral import TheoremSpectralTopology
        adj = np.zeros((10, 10))
        for i in range(9):
            adj[i, i+1] = 1; adj[i+1, i] = 1
        tst = TheoremSpectralTopology(adj)
        report = tst.connectivity_report()
        assert "n_theorems" in report
        assert report["n_theorems"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
