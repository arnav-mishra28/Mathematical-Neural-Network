"""
Example 31 — Geometric Transformer Architectures: Full Pipeline
Demonstrates all 8 parts: geometric tokens, geodesic attention, spectral
positional encodings, categorical attention, quantum attention, PDE-aware
transformers, graph transformers, and hierarchical reasoning.
"""
import numpy as np
import torch


def main():
    print("=" * 70)
    print("  GEOMETRIC TRANSFORMER ARCHITECTURES — Full Pipeline")
    print("=" * 70)

    # ---- Part 1: Geometric Tokens ----
    print("\n[Part 1] Geometric Token Representation")
    from mnn.geometric_transformer.tokens import GeometricToken, TokenType, ManifoldEmbedding

    t1 = GeometricToken(np.array([1, 0, 0.0]), TokenType.THEOREM, curvature=0)
    t2 = GeometricToken(np.array([0, 1, 0.0]), TokenType.OPERATOR, curvature=0)
    t3 = GeometricToken(np.array([1, 0, 0.0]), TokenType.THEOREM, curvature=1.0)
    t4 = GeometricToken(np.array([0, 1, 0.0]), TokenType.THEOREM, curvature=1.0)

    print(f"  Euclidean dist(t1,t2) = {t1.geodesic_distance(t2):.4f}")
    print(f"  Spherical dist(t3,t4) = {t3.geodesic_distance(t4):.4f}")

    emb = ManifoldEmbedding(100, 32, curvature=0.5)
    ids = torch.randint(0, 100, (4, 16))
    embedded = emb(ids)
    print(f"  ManifoldEmbedding: {ids.shape} -> {embedded.shape}")

    # ---- Part 2: Geometric Attention ----
    print("\n[Part 2] Geometric Attention")
    from mnn.geometric_transformer.tokens import GeometricAttention

    geo_attn = GeometricAttention(32, n_heads=4, curvature=0.5)
    x = torch.randn(4, 16, 32)
    out = geo_attn(x)
    print(f"  GeometricAttention: {x.shape} -> {out.shape}")
    print(f"  Learnable curvature: {geo_attn.curvature.item():.4f}")

    # ---- Part 3: Spectral Positional Encodings ----
    print("\n[Part 3] Manifold Positional Encodings")
    from mnn.geometric_transformer.positional import SpectralPositionalEncoding, RandomWalkEncoding

    spe = SpectralPositionalEncoding(32, n_eigvecs=8)
    # Create a chain graph
    adj = torch.zeros(2, 10, 10)
    for i in range(9):
        adj[:, i, i+1] = 1
        adj[:, i+1, i] = 1
    x_graph = torch.randn(2, 10, 32)
    out_spe = spe(x_graph, adj)
    print(f"  SpectralPE: {x_graph.shape} -> {out_spe.shape}")

    rwe = RandomWalkEncoding(32, walk_length=4)
    out_rw = rwe(x_graph, adj)
    print(f"  RandomWalkPE: {x_graph.shape} -> {out_rw.shape}")

    # ---- Part 4: Categorical Attention ----
    print("\n[Part 4] Category-Theoretic Attention")
    from mnn.geometric_transformer.positional import CategoricalAttention

    cat_attn = CategoricalAttention(32, n_heads=4)
    x = torch.randn(4, 8, 32)
    out = cat_attn(x)
    print(f"  CategoricalAttention: {x.shape} -> {out.shape}")
    print(f"  (Morphism-based compositional scoring)")

    # ---- Part 5: Quantum Geometric Attention ----
    print("\n[Part 5] Quantum Geometric Attention")
    from mnn.geometric_transformer.quantum_attention import QuantumGeometricAttention

    q_attn = QuantumGeometricAttention(32, n_heads=4)
    x = torch.randn(4, 8, 32)
    out = q_attn(x)
    print(f"  QuantumAttention: {x.shape} -> {out.shape}")
    print(f"  Phase gates: {q_attn.phase_gate.data}")

    # ---- Part 6: PDE-Aware Attention ----
    print("\n[Part 6] PDE-Aware Attention")
    from mnn.geometric_transformer.quantum_attention import PDEAwareAttention

    pde_attn = PDEAwareAttention(32, n_heads=4)
    x = torch.randn(4, 32, 32)
    out = pde_attn(x)
    print(f"  PDEAwareAttention: {x.shape} -> {out.shape}")
    print(f"  (Gradient + Laplacian enriched)")

    # ---- Part 7: Theorem Graph Transformer ----
    print("\n[Part 7] Theorem Graph Transformer")
    from mnn.geometric_transformer.graph_transformer import TheoremGraphTransformer

    gt = TheoremGraphTransformer(embed_dim=32, n_heads=4, n_layers=3, n_edge_types=6)
    print(f"  GraphTransformer params: {gt.count_parameters():,}")

    # Create a theorem graph
    n_theorems = 12
    x = torch.randn(2, n_theorems, 32)
    adj = torch.zeros(2, n_theorems, n_theorems)
    # Chain: 0->1->2->...->11
    for i in range(n_theorems - 1):
        adj[:, i, i+1] = 1
        adj[:, i+1, i] = 1
    # Cross-links
    adj[:, 0, 5] = 1; adj[:, 5, 0] = 1
    adj[:, 3, 8] = 1; adj[:, 8, 3] = 1

    edge_types = torch.randint(0, 6, (2, n_theorems, n_theorems))
    out = gt(x, adj, edge_types)
    print(f"  Input: {x.shape}, Adj: {adj.shape}")
    print(f"  Output: {out.shape}")

    # ---- Part 8: Hierarchical Geometric Reasoning ----
    print("\n[Part 8] Hierarchical Geometric Reasoning")
    from mnn.geometric_transformer.graph_transformer import HierarchicalGeometricTransformer

    hgt = HierarchicalGeometricTransformer(
        embed_dim=32, n_heads=4, n_levels=3, pool_ratio=0.5)
    print(f"  HierarchicalTransformer params: {hgt.count_parameters():,}")

    x = torch.randn(2, 16, 32)
    adj = (torch.rand(2, 16, 16) > 0.5).float()
    result = hgt(x, adj)

    print(f"  Input: {x.shape}")
    for i, level in enumerate(result["levels"]):
        labels = ["Equations (local)", "Proofs (mid)", "Theorem networks (large)"]
        print(f"  Level {i} ({labels[i]}): {level.shape}")
    print(f"  Finest output: {result['finest'].shape}")

    print("\n" + "=" * 70)
    print("  GEOMETRIC TRANSFORMER ARCHITECTURES — All 8 parts complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
