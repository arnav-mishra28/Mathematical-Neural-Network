"""
Example 30 — Quantum PDE Solvers: Full Pipeline
Demonstrates all 8 parts: quantum states, operators, neural operators,
quantum evolution, spectral solving, geometric regularization,
PDE discovery, and category-theoretic structure.
"""
import numpy as np


def main():
    print("=" * 70)
    print("  QUANTUM PDE SOLVERS — Full Pipeline")
    print("=" * 70)

    # ---- Parts 1-2: Quantum States + Operators ----
    print("\n[Parts 1-2] Quantum State PDE Representation + Operators")
    from mnn.quantum_pde.states import QuantumPDEState, PDEOperator

    grid = np.linspace(0, 2*np.pi, 128, endpoint=False)
    dx = grid[1] - grid[0]

    psi = QuantumPDEState.gaussian_packet(grid, center=np.pi, width=0.5, k0=3.0)
    print(f"  {psi}")
    print(f"  Energy: {psi.total_energy():.4f}")

    lap = PDEOperator.laplacian_1d(128, dx)
    print(f"  {lap}")
    evals, _ = lap.eigendecomposition()
    print(f"  Laplacian eigenvalue range: [{evals.min():.2f}, {evals.max():.2f}]")

    # Evolve with Laplacian (diffusion)
    propagator = lap.propagator(dt=0.001)
    psi_evolved = psi.evolve(propagator, 0.001)
    print(f"  After evolution: {psi_evolved}")

    # ---- Parts 3-4: Neural Operator + Quantum Evolution ----
    print("\n[Parts 3-4] Neural Operator + Quantum Evolution")
    import torch
    from mnn.quantum_pde.neural_operator import NeuralPDEOperator, QuantumPDENet

    op = NeuralPDEOperator(1, 1, width=32, n_layers=3, n_grid=64)
    print(f"  NeuralPDEOperator params: {op.count_parameters():,}")
    x_in = torch.randn(8, 64, 1)
    out = op(x_in)
    print(f"  Input: {x_in.shape} -> Output: {out.shape}")

    qpde = QuantumPDENet(n_grid=64, width=16, n_operator_layers=2, n_evolution_steps=2)
    out2 = qpde(x_in)
    print(f"  QuantumPDENet params: {qpde.count_parameters():,}, output: {out2.shape}")

    # ---- Part 5: Spectral PDE Solving ----
    print("\n[Part 5] Spectral PDE Solving")
    from mnn.quantum_pde.spectral import SpectralPDESolver

    solver = SpectralPDESolver(128)

    # Heat equation
    u0 = np.sin(solver.x)
    heat_traj = solver.solve_heat(u0, alpha=0.1, dt=0.01, n_steps=50)
    print(f"  Heat eq: initial max={np.max(np.abs(u0)):.4f}, "
          f"final max={np.max(np.abs(heat_traj[-1])):.4f}")

    # Wave equation
    v0 = np.zeros(128)
    wave_traj = solver.solve_wave(u0, v0, c=1.0, dt=0.01, n_steps=50)
    print(f"  Wave eq: {wave_traj.shape[0]} timesteps, "
          f"final max={np.max(np.abs(wave_traj[-1])):.4f}")

    # Schrödinger
    psi0 = np.exp(-(solver.x - np.pi)**2).astype(complex)
    psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * solver.dx)
    V = 0.5 * (solver.x - np.pi)**2  # harmonic trap
    schro_traj = solver.solve_schrodinger(psi0, V, dt=0.01, n_steps=20)
    norm_init = np.sum(np.abs(schro_traj[0])**2) * solver.dx
    norm_final = np.sum(np.abs(schro_traj[-1])**2) * solver.dx
    print(f"  Schrödinger: norm init={norm_init:.4f}, final={norm_final:.4f}")

    # ---- Part 6: Geometric Regularization ----
    print("\n[Part 6] Geometric Regularization")
    from mnn.quantum_pde.spectral import GeometricPDERegularizer

    reg = GeometricPDERegularizer(norm_weight=1.0, smooth_weight=0.5, curvature_weight=0.1)
    u_test = torch.randn(4, 64, 1)
    loss = reg(u_test)
    print(f"  Geometric loss: {loss.item():.4f}")
    print(f"    Norm: {reg.norm_preservation_loss(u_test).item():.4f}")
    print(f"    Smooth: {reg.smoothness_loss(u_test).item():.4f}")
    print(f"    Curvature: {reg.curvature_loss(u_test).item():.4f}")

    # ---- Part 7: PDE Discovery ----
    print("\n[Part 7] PDE Discovery")
    from mnn.quantum_pde.pde_discovery import PDEDiscoveryEngine

    # Generate heat equation data
    alpha_true = 0.5
    nx, nt = 64, 200
    dx_d = 2 * np.pi / nx
    dt_d = 0.001
    x_d = np.linspace(0, 2*np.pi, nx, endpoint=False)
    k_d = np.fft.fftfreq(nx, d=dx_d) * 2 * np.pi
    u0_d = np.sin(x_d) + 0.5*np.sin(3*x_d)
    u_data = np.zeros((nt, nx))
    u_hat = np.fft.fft(u0_d)
    for t in range(nt):
        u_data[t] = np.real(np.fft.ifft(u_hat))
        u_hat *= np.exp(-alpha_true * k_d**2 * dt_d)

    engine = PDEDiscoveryEngine(poly_order=2, deriv_order=3, threshold=0.01)
    result = engine.discover(u_data, dx_d, dt_d, verbose=True)
    print(f"  Discovered equation: {result['equation']}")
    print(f"  True: u_t = {alpha_true}*u_xx")

    # ---- Part 8: Category-Theoretic PDE ----
    print("\n[Part 8] Category-Theoretic PDE Structure")
    from mnn.quantum_pde.categorical_pde import (
        PDECategory, SolutionSpace, PDEMorphism, DiscretizationFunctor,
    )

    cat = PDECategory("DiffusionCategory")
    cat.add_space(SolutionSpace("initial", 64, regularity="C^inf"))
    cat.add_space(SolutionSpace("evolved", 64, regularity="C^inf"))
    cat.add_space(SolutionSpace("spectral", 64, regularity="L2"))

    # Operators as morphisms
    functor = DiscretizationFunctor(64, (0, 2*np.pi))
    lap_mat = functor.discretize_laplacian()
    from scipy.linalg import expm
    heat_prop = expm(alpha_true * lap_mat * 0.1)

    cat.add_morphism(PDEMorphism("heat_evolve", "initial", "evolved",
                                  lambda u: heat_prop @ u))
    cat.add_morphism(PDEMorphism("to_spectral", "initial", "spectral",
                                  lambda u: np.fft.fft(u)))
    cat.add_morphism(PDEMorphism("evolve_spectral", "spectral", "spectral",
                                  lambda u_hat: u_hat * np.exp(-alpha_true * (np.fft.fftfreq(64, d=2*np.pi/64)*2*np.pi)**2 * 0.1)))

    print(f"  {cat.summary()}")

    # Verify functor
    verification = functor.verify_functor(lap_mat, functor.discretize_gradient(),
                                           np.random.randn(64))
    print(f"  Functor composition preserved: {verification['composition_preserved']}")

    print("\n" + "=" * 70)
    print("  QUANTUM PDE SOLVERS — All 8 parts complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
