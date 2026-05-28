"""Tests for Geometric Transformer Architectures (mnn.geometric_transformer)."""
import numpy as np
import torch
import pytest


# ===== Part 1: Geometric Tokens =====

class TestGeometricToken:
    def test_euclidean_distance(self):
        from mnn.geometric_transformer.tokens import GeometricToken, TokenType
        t1 = GeometricToken(np.array([1, 0, 0.0]), TokenType.THEOREM, curvature=0)
        t2 = GeometricToken(np.array([0, 1, 0.0]), TokenType.THEOREM, curvature=0)
        d = t1.geodesic_distance(t2)
        assert abs(d - np.sqrt(2)) < 1e-6

    def test_spherical_distance(self):
        from mnn.geometric_transformer.tokens import GeometricToken, TokenType
        t1 = GeometricToken(np.array([1, 0, 0.0]), TokenType.THEOREM, curvature=1.0)
        t2 = GeometricToken(np.array([0, 1, 0.0]), TokenType.THEOREM, curvature=1.0)
        d = t1.geodesic_distance(t2)
        assert abs(d - np.pi/2) < 1e-6

    def test_exponential_map(self):
        from mnn.geometric_transformer.tokens import GeometricToken
        t = GeometricToken(np.array([1.0, 0, 0]), curvature=0)
        t2 = t.exponential_map(np.array([0, 1.0, 0]))
        assert t2.embedding[1] == 1.0

    def test_log_map(self):
        from mnn.geometric_transformer.tokens import GeometricToken
        t1 = GeometricToken(np.array([0.0, 0, 0]))
        t2 = GeometricToken(np.array([1.0, 1, 0]))
        v = t1.logarithmic_map(t2)
        assert np.allclose(v, [1, 1, 0])


class TestManifoldEmbedding:
    def test_forward(self):
        from mnn.geometric_transformer.tokens import ManifoldEmbedding
        emb = ManifoldEmbedding(100, 32, curvature=0)
        ids = torch.randint(0, 100, (4, 16))
        out = emb(ids)
        assert out.shape == (4, 16, 32)

    def test_curved(self):
        from mnn.geometric_transformer.tokens import ManifoldEmbedding
        emb = ManifoldEmbedding(100, 32, curvature=1.0)
        ids = torch.randint(0, 100, (4, 16))
        out = emb(ids)
        assert out.shape == (4, 16, 32)


# ===== Part 2: Geometric Attention =====

class TestGeometricAttention:
    def test_forward(self):
        from mnn.geometric_transformer.tokens import GeometricAttention
        attn = GeometricAttention(32, n_heads=4)
        x = torch.randn(4, 16, 32)
        out = attn(x)
        assert out.shape == (4, 16, 32)

    def test_with_curvature(self):
        from mnn.geometric_transformer.tokens import GeometricAttention
        attn = GeometricAttention(32, n_heads=4, curvature=0.5)
        x = torch.randn(4, 8, 32)
        out = attn(x)
        assert out.shape == (4, 8, 32)


# ===== Part 3: Positional Encodings =====

class TestPositionalEncodings:
    def test_spectral_sequential(self):
        from mnn.geometric_transformer.positional import SpectralPositionalEncoding
        pe = SpectralPositionalEncoding(32, max_nodes=64)
        x = torch.randn(4, 16, 32)
        out = pe(x)
        assert out.shape == (4, 16, 32)

    def test_spectral_with_adjacency(self):
        from mnn.geometric_transformer.positional import SpectralPositionalEncoding
        pe = SpectralPositionalEncoding(32, n_eigvecs=8)
        x = torch.randn(2, 10, 32)
        adj = torch.zeros(2, 10, 10)
        for b in range(2):
            for i in range(9):
                adj[b, i, i+1] = 1
                adj[b, i+1, i] = 1
        out = pe(x, adj)
        assert out.shape == (2, 10, 32)

    def test_random_walk(self):
        from mnn.geometric_transformer.positional import RandomWalkEncoding
        rw = RandomWalkEncoding(32, walk_length=4)
        x = torch.randn(2, 8, 32)
        adj = (torch.rand(2, 8, 8) > 0.5).float()
        out = rw(x, adj)
        assert out.shape == (2, 8, 32)


# ===== Part 4: Categorical Attention =====

class TestCategoricalAttention:
    def test_forward(self):
        from mnn.geometric_transformer.positional import CategoricalAttention
        attn = CategoricalAttention(32, n_heads=4)
        x = torch.randn(4, 8, 32)
        out = attn(x)
        assert out.shape == (4, 8, 32)


# ===== Part 5: Quantum Geometric Attention =====

class TestQuantumAttention:
    def test_forward(self):
        from mnn.geometric_transformer.quantum_attention import QuantumGeometricAttention
        attn = QuantumGeometricAttention(32, n_heads=4)
        x = torch.randn(4, 8, 32)
        out = attn(x)
        assert out.shape == (4, 8, 32)

    def test_phase_gate(self):
        from mnn.geometric_transformer.quantum_attention import QuantumGeometricAttention
        attn = QuantumGeometricAttention(32, n_heads=4)
        assert attn.phase_gate.shape == (4,)


# ===== Part 6: PDE-Aware Attention =====

class TestPDEAttention:
    def test_forward(self):
        from mnn.geometric_transformer.quantum_attention import PDEAwareAttention
        attn = PDEAwareAttention(32, n_heads=4)
        x = torch.randn(4, 16, 32)
        out = attn(x)
        assert out.shape == (4, 16, 32)


# ===== Part 7: Graph Transformer =====

class TestGraphTransformer:
    def test_graph_attention(self):
        from mnn.geometric_transformer.graph_transformer import GraphAttentionLayer
        attn = GraphAttentionLayer(32, n_heads=4, n_edge_types=6)
        x = torch.randn(2, 10, 32)
        adj = (torch.rand(2, 10, 10) > 0.5).float()
        out = attn(x, adj)
        assert out.shape == (2, 10, 32)

    def test_with_edge_types(self):
        from mnn.geometric_transformer.graph_transformer import GraphAttentionLayer
        attn = GraphAttentionLayer(32, n_heads=4, n_edge_types=6)
        x = torch.randn(2, 8, 32)
        adj = (torch.rand(2, 8, 8) > 0.5).float()
        et = torch.randint(0, 6, (2, 8, 8))
        out = attn(x, adj, et)
        assert out.shape == (2, 8, 32)

    def test_full_transformer(self):
        from mnn.geometric_transformer.graph_transformer import TheoremGraphTransformer
        model = TheoremGraphTransformer(embed_dim=32, n_heads=4, n_layers=2)
        x = torch.randn(2, 10, 32)
        adj = (torch.rand(2, 10, 10) > 0.5).float()
        out = model(x, adj)
        assert out.shape == (2, 10, 32)


# ===== Part 8: Hierarchical Reasoning =====

class TestHierarchical:
    def test_pooling(self):
        from mnn.geometric_transformer.graph_transformer import HierarchicalPooling
        pool = HierarchicalPooling(32, pool_ratio=0.5)
        x = torch.randn(2, 10, 32)
        adj = (torch.rand(2, 10, 10) > 0.5).float()
        px, pa, pi = pool(x, adj)
        assert px.shape[1] == 5
        assert pa.shape == (2, 5, 5)

    def test_hierarchical_transformer(self):
        from mnn.geometric_transformer.graph_transformer import HierarchicalGeometricTransformer
        model = HierarchicalGeometricTransformer(
            embed_dim=32, n_heads=4, n_levels=3, pool_ratio=0.5)
        x = torch.randn(2, 16, 32)
        adj = (torch.rand(2, 16, 16) > 0.5).float()
        result = model(x, adj)
        assert result["finest"].shape == (2, 16, 32)
        assert len(result["levels"]) == 3
        # Each level should be smaller
        for i in range(1, len(result["levels"])):
            assert result["levels"][i].shape[1] <= result["levels"][i-1].shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
