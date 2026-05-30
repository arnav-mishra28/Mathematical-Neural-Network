"""
Example 32 — Spectral Manifold Learning: Full Pipeline
Demonstrates all 9 parts: Laplacian, spectral decomposition, graph analysis,
spectral embeddings, manifold harmonics, spectral attention, quantum spectral,
PDE spectral solving, and theorem spectral topology.
"""
import numpy as np
import torch


def main():
    print("=" * 70)
    print("  SPECTRAL MANIFOLD LEARNING — Full Pipeline")
    print("=" * 70)

    # ---- Parts 1-2: Laplacian + Spectral Decomposition ----
    print("\n[Parts 1-2] Manifold Laplacian + Spectral Decomposition")
    from mnn.spectral_manifold.laplacian import ManifoldLaplacian, SpectralDecomposition

    # Create a manifold (circle-like point cloud)
    t = np.linspace(0, 2*np.pi, 50, endpoint=False)
    pts = np.stack([np.cos(t), np.sin(t)], axis=1) + np.random.randn(50, 2) * 0.1

    lap = ManifoldLaplacian(pts, k_neighbors=8)
    spec = SpectralDecomposition(lap, 20)
    print(f"  {spec.summary()}")
    print(f"  Frequencies: {spec.frequencies[:5]}")
    print(f"  Spectral gap: {spec.spectral_gap():.4f}")

    # ---- Part 3: Graph Spectral Analysis ----
    print("\n[Part 3] Graph Spectral Analysis")
    from mnn.spectral_manifold.embeddings import GraphSpectralAnalyzer

    # Create a graph with 2 clusters
    adj = np.zeros((20, 20))
    for i in range(9):
        adj[i, i+1] = 1; adj[i+1, i] = 1
    for i in range(10, 19):
        adj[i, i+1] = 1; adj[i+1, i] = 1
    adj[9, 10] = 1; adj[10, 9] = 1  # bridge

    analyzer = GraphSpectralAnalyzer(adj)
    print(f"  {analyzer.summary()}")
    communities = analyzer.detect_communities(2)
    for c, info in communities.items():
        print(f"  Community {c}: {info['size']} members, density={info['internal_density']:.3f}")
    bottlenecks = analyzer.bottleneck_nodes(3)
    print(f"  Bottleneck nodes: {bottlenecks}")

    # ---- Part 4: Spectral Embeddings ----
    print("\n[Part 4] Spectral Embeddings")
    from mnn.spectral_manifold.embeddings import SpectralEmbedding

    se = SpectralEmbedding(n_components=4)
    coords = se.fit_transform(adj)
    print(f"  Embedded {adj.shape[0]} nodes into {coords.shape[1]}D spectral space")
    print(f"  Coord range: [{coords.min():.3f}, {coords.max():.3f}]")

    # ---- Part 5: Manifold Harmonics ----
    print("\n[Part 5] Manifold Harmonics")
    from mnn.spectral_manifold.embeddings import ManifoldHarmonics

    mh = ManifoldHarmonics(spec)
    signal = np.sin(t) + 0.3 * np.random.randn(50)
    result = mh.decompose_signal(signal)
    print(f"  Signal energy: total={result['energy_total']:.3f}")
    print(f"    Low freq:  {result['energy_low']:.3f}")
    print(f"    Mid freq:  {result['energy_mid']:.3f}")
    print(f"    High freq: {result['energy_high']:.3f}")
    freq_resp = mh.frequency_response()
    print(f"  Bandwidth: {freq_resp['bandwidth']:.3f}, "
          f"Effective dim: {freq_resp['effective_dimension']}")

    # ---- Part 6: Spectral Attention ----
    print("\n[Part 6] Spectral Attention")
    from mnn.spectral_manifold.spectral_attention import SpectralAttention

    s_attn = SpectralAttention(32, n_harmonics=8, n_heads=4)
    x = torch.randn(4, 16, 32)
    out = s_attn(x)
    print(f"  SpectralAttention: {x.shape} -> {out.shape}")
    print(f"  Learnable freq filter shape: {s_attn.freq_filter.shape}")

    # ---- Part 7: Quantum Spectral Geometry ----
    print("\n[Part 7] Quantum Spectral Geometry")
    from mnn.spectral_manifold.spectral_attention import QuantumSpectralLayer

    qsl = QuantumSpectralLayer(16, n_harmonics=8)
    r, i = torch.randn(4, 16), torch.randn(4, 16)
    out_r, out_i = qsl(r, i)
    norms = torch.sqrt((out_r**2 + out_i**2).sum(dim=-1))
    print(f"  Input: ({r.shape}, {i.shape})")
    print(f"  Output norms (should ≈ 1): {norms.data}")
    energy = qsl.spectral_energy(r, i)
    print(f"  Spectral energy distribution: {energy[0].data[:4]}...")

    # ---- Part 8: PDE Spectral Solving ----
    print("\n[Part 8] PDE Spectral Solving")
    from mnn.spectral_manifold.pde_spectral import SpectralPDEEvolver

    chain_adj = np.zeros((30, 30))
    for j in range(29):
        chain_adj[j, j+1] = 1; chain_adj[j+1, j] = 1
    chain_lap = ManifoldLaplacian.from_adjacency(chain_adj)
    chain_spec = SpectralDecomposition(chain_lap, 20)
    evolver = SpectralPDEEvolver(chain_spec)

    u0 = np.zeros(30); u0[15] = 1.0
    heat_traj = evolver.solve_heat(u0, alpha=0.5, dt=0.1, n_steps=20)
    print(f"  Heat: initial max={np.max(u0):.3f}, "
          f"final max={np.max(np.abs(heat_traj[-1])):.3f}")

    comp = evolver.operator_compression(5)
    print(f"  Compression: {comp['n_modes']} modes capture "
          f"{comp['energy_captured']*100:.1f}% energy")

    # ---- Part 9: Theorem Spectral Topology ----
    print("\n[Part 9] Theorem Spectral Topology")
    from mnn.spectral_manifold.pde_spectral import TheoremSpectralTopology

    names = [f"Thm_{i}" for i in range(20)]
    tst = TheoremSpectralTopology(adj, names)
    report = tst.connectivity_report()
    print(f"  Theorems: {report['n_theorems']}")
    print(f"  Algebraic connectivity: {report['algebraic_connectivity']:.4f}")
    print(f"  Spectral gap: {report['spectral_gap']:.4f}")
    print(f"  Connected components: {report['n_connected_components']}")

    clusters = tst.theorem_clusters(2)
    for c, members in clusters.items():
        print(f"  Cluster {c}: {members[:5]}{'...' if len(members)>5 else ''}")

    bottlenecks = tst.proof_bottlenecks(3)
    for b in bottlenecks:
        print(f"  Bottleneck: {b['node']} (score={b['bottleneck_score']:.2f})")

    importance = tst.spectral_importance()[:5]
    print(f"  Most important: {[f'{i['node']}({i['importance']:.3f})' for i in importance]}")

    print("\n" + "=" * 70)
    print("  SPECTRAL MANIFOLD LEARNING — All 9 parts complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
