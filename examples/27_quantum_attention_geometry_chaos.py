"""
Example 27 — Quantum-Inspired Neural Math: Attention, Geometry, Chaos
Parts 4-6: Quantum attention, geometric learning, quantum chaos.
"""
import numpy as np


def main():
    print("=" * 65)
    print("  QUANTUM-INSPIRED NEURAL MATH — Parts 4-6")
    print("=" * 65)

    # ---- Part 4: Quantum Attention ----
    print("\n[4] Quantum-Inspired Attention")
    import torch
    from mnn.quantum.attention import QuantumMultiHeadAttention, QuantumTransformer

    # Test quantum attention
    d_model = 16
    qmha = QuantumMultiHeadAttention(d_model, n_heads=4)
    x = torch.randn(2, 5, d_model)  # batch=2, seq=5
    out, weights = qmha(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    print(f"  Attention heads: {len(weights)}, weights shape: {weights[0].shape}")

    # Quantum transformer
    qt = QuantumTransformer(8, 2, d_model=32, n_heads=4, n_layers=2)
    x_in = torch.randn(10, 8)
    y_out, all_w = qt(x_in)
    print(f"  QuantumTransformer(8->2): {qt.count_parameters():,} params")
    print(f"  Output shape: {y_out.shape}")

    # ---- Part 5: Quantum Geometric Learning ----
    print("\n[5] Quantum Geometric Learning")
    from mnn.quantum.geometric import (
        FubiniStudyMetric, QuantumEmbedding, QuantumGeometricNetwork,
    )

    # Fubini-Study distances
    psi1 = np.array([1, 0], dtype=complex)
    psi2 = np.array([1, 1], dtype=complex) / np.sqrt(2)
    psi3 = np.array([0, 1], dtype=complex)
    print(f"  FS dist(|0>, |+>) = {FubiniStudyMetric.distance(psi1, psi2):.4f}")
    print(f"  FS dist(|0>, |1>) = {FubiniStudyMetric.distance(psi1, psi3):.4f} (= pi/2)")

    # Berry phase around a loop
    n_pts = 20
    loop = [np.array([np.cos(t), np.sin(t)*np.exp(1j*t)]) for t in np.linspace(0, 2*np.pi, n_pts)]
    bp = FubiniStudyMetric.berry_phase(loop)
    print(f"  Berry phase: {bp:.4f}")

    # Metric tensor
    g = FubiniStudyMetric.metric_tensor(np.array([1, 1j]) / np.sqrt(2))
    print(f"  Fubini-Study metric tensor:\n    {np.round(g, 4)}")

    # Quantum geometric network
    qgn = QuantumGeometricNetwork(4, 1, state_dim=16, n_geo_layers=3)
    x_test = torch.randn(50, 4)
    pred = qgn(x_test)
    print(f"  QGN(4->1): {qgn.count_parameters():,} params, output={pred.shape}")

    # Geodesic distances through layers
    dists = qgn.geodesic_distances(x_test[:5])
    print(f"  Geodesic distances per layer: {[f'{d:.4f}' for d in dists]}")

    # ---- Part 6: Quantum Chaos ----
    print("\n[6] Quantum Chaos")
    from mnn.quantum.chaos import (
        RandomMatrixEnsemble, SpectralAnalyzer,
        QuantumKickedTop, QuantumLyapunov, QuantumEntanglementDynamics,
    )

    # GOE vs Poisson comparison
    print("\n  Spectral statistics (n=200):")
    goe = RandomMatrixEnsemble.goe(200, seed=42)
    evals_goe = SpectralAnalyzer.eigenvalues(goe)
    stats_goe = SpectralAnalyzer.classify_dynamics(evals_goe)
    print(f"    GOE: {stats_goe['classification']}, <r>={stats_goe['mean_r']:.3f}")

    # Diagonal (integrable) system
    diag = np.diag(np.random.default_rng(42).standard_normal(200))
    evals_diag = SpectralAnalyzer.eigenvalues(diag)
    stats_diag = SpectralAnalyzer.classify_dynamics(evals_diag)
    print(f"    Diagonal: {stats_diag['classification']}, <r>={stats_diag['mean_r']:.3f}")

    # GUE
    gue = RandomMatrixEnsemble.gue(200, seed=42)
    evals_gue = SpectralAnalyzer.eigenvalues(gue)
    stats_gue = SpectralAnalyzer.classify_dynamics(evals_gue)
    print(f"    GUE: {stats_gue['classification']}, <r>={stats_gue['mean_r']:.3f}")

    # Quantum kicked top
    print("\n  Quantum Kicked Top (j=8, k=3.0):")
    qkt = QuantumKickedTop(j=8, k=3.0)
    qkt_stats = qkt.spectral_statistics()
    print(f"    Floquet: {qkt_stats['classification']}, <r>={qkt_stats['mean_r']:.3f}")

    psi0 = np.zeros(qkt.dim, dtype=complex)
    psi0[0] = 1.0
    traj = qkt.evolve(psi0, n_kicks=50)
    print(f"    Evolved for {len(traj)-1} kicks, final overlap with init: "
          f"{np.abs(np.vdot(traj[0], traj[-1]))**2:.6f}")

    # Entanglement growth
    print("\n  Entanglement dynamics:")
    n_q = 4
    dim = 2**n_q
    H_chaos = RandomMatrixEnsemble.goe(dim, seed=123)
    psi_init = np.zeros(dim, dtype=complex)
    psi_init[0] = 1.0
    times = np.linspace(0, 5, 30)
    d1, d2 = 2**(n_q//2), 2**(n_q - n_q//2)
    S = QuantumEntanglementDynamics.entanglement_entropy_evolution(
        H_chaos, psi_init, (d1, d2), times)
    growth = QuantumEntanglementDynamics.classify_entanglement_growth(S, times)
    page = QuantumEntanglementDynamics.page_entropy(d1, d2)
    print(f"    Growth: {growth['growth_type']}")
    print(f"    Max entropy: {growth['max_entropy']:.3f}, Page value: {page:.3f}")

    print("\n" + "=" * 65)
    print("  QUANTUM-INSPIRED NEURAL MATH — All 6 parts complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
