"""Tests for Quantum PDE Solvers (mnn.quantum_pde)."""
import numpy as np
import torch
import pytest


# ===== Parts 1-2: States + Operators =====

class TestQuantumPDEState:
    def test_from_real_field(self):
        from mnn.quantum_pde.states import QuantumPDEState
        grid = np.linspace(0, 1, 50)
        u = np.sin(2 * np.pi * grid)
        state = QuantumPDEState.from_real_field(u, grid)
        assert state.n_grid == 50
        assert state.norm() > 0

    def test_gaussian_packet(self):
        from mnn.quantum_pde.states import QuantumPDEState
        grid = np.linspace(-5, 5, 100)
        psi = QuantumPDEState.gaussian_packet(grid, center=0, width=1.0)
        assert abs(psi.norm() - 1.0) < 0.1  # approximately normalized

    def test_inner_product_self(self):
        from mnn.quantum_pde.states import QuantumPDEState
        grid = np.linspace(0, 1, 64)
        u = np.ones(64, dtype=complex)
        s = QuantumPDEState(u, grid)
        ip = s.inner_product(s)
        assert ip.real > 0

    def test_probability_density(self):
        from mnn.quantum_pde.states import QuantumPDEState
        grid = np.linspace(0, 1, 32)
        psi = QuantumPDEState.gaussian_packet(grid, center=0.5, width=0.1)
        pd = psi.probability_density()
        assert pd.shape == (32,)
        assert np.all(pd >= 0)


class TestPDEOperator:
    def test_laplacian(self):
        from mnn.quantum_pde.states import PDEOperator
        lap = PDEOperator.laplacian_1d(64, 0.01)
        assert lap.matrix.shape == (64, 64)
        assert lap.hermitian  # should be symmetric

    def test_first_derivative(self):
        from mnn.quantum_pde.states import PDEOperator
        d1 = PDEOperator.first_derivative_1d(64, 0.01)
        assert d1.matrix.shape == (64, 64)

    def test_eigendecomposition(self):
        from mnn.quantum_pde.states import PDEOperator
        lap = PDEOperator.laplacian_1d(32, 0.1)
        evals, evecs = lap.eigendecomposition()
        assert len(evals) == 32
        assert np.all(evals <= 1e-10)  # Laplacian has non-positive eigenvalues

    def test_combine(self):
        from mnn.quantum_pde.states import PDEOperator
        lap = PDEOperator.laplacian_1d(32, 0.1)
        ident = PDEOperator.identity(32, 0.1)
        combined = lap.combine(ident, 0.5, 1.0)
        assert combined.matrix.shape == (32, 32)


# ===== Parts 3-4: Neural Operator + Quantum Evolution =====

class TestNeuralOperator:
    def test_forward_shape(self):
        from mnn.quantum_pde.neural_operator import NeuralPDEOperator
        op = NeuralPDEOperator(in_channels=1, out_channels=1, width=16, n_layers=2, n_grid=32)
        x = torch.randn(4, 32, 1)
        out = op(x)
        assert out.shape == (4, 32, 1)

    def test_quantum_evolution_unitarity(self):
        from mnn.quantum_pde.neural_operator import QuantumEvolutionLayer
        evo = QuantumEvolutionLayer(16)
        assert evo.unitarity_error() < 0.05  # random init, not trained yet

    def test_quantum_pde_net(self):
        from mnn.quantum_pde.neural_operator import QuantumPDENet
        net = QuantumPDENet(n_grid=32, in_channels=1, out_channels=1, width=16,
                            n_operator_layers=2, n_evolution_steps=1)
        x = torch.randn(4, 32, 1)
        out = net(x)
        assert out.shape == (4, 32, 1)


# ===== Parts 5-6: Spectral + Geometric =====

class TestSpectralSolver:
    def test_derivative(self):
        from mnn.quantum_pde.spectral import SpectralPDESolver
        solver = SpectralPDESolver(128)
        u = np.sin(solver.x)
        du = solver.derivative(u, 1)
        expected = np.cos(solver.x)
        assert np.allclose(du, expected, atol=0.05)

    def test_heat_equation(self):
        from mnn.quantum_pde.spectral import SpectralPDESolver
        solver = SpectralPDESolver(64)
        u0 = np.sin(solver.x)
        traj = solver.solve_heat(u0, alpha=0.1, dt=0.01, n_steps=10)
        assert traj.shape[0] == 11
        # Heat equation: amplitude should decay
        assert np.max(np.abs(traj[-1])) < np.max(np.abs(traj[0]))

    def test_schrodinger(self):
        from mnn.quantum_pde.spectral import SpectralPDESolver
        solver = SpectralPDESolver(64)
        psi0 = np.exp(-(solver.x - np.pi)**2).astype(complex)
        psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * solver.dx)
        V = np.zeros(64)
        traj = solver.solve_schrodinger(psi0, V, dt=0.01, n_steps=5)
        # Norm should be approximately preserved
        norm_final = np.sum(np.abs(traj[-1])**2) * solver.dx
        norm_init = np.sum(np.abs(traj[0])**2) * solver.dx
        assert abs(norm_final - norm_init) < 0.1

    def test_power_spectrum(self):
        from mnn.quantum_pde.spectral import SpectralPDESolver
        solver = SpectralPDESolver(64)
        u = np.sin(solver.x)
        k, power = solver.power_spectrum(u)
        assert len(k) == 64


class TestSpectralNet:
    def test_forward(self):
        from mnn.quantum_pde.spectral import SpectralPDENet
        net = SpectralPDENet(in_channels=1, out_channels=1, width=16, n_layers=2, n_modes=8)
        x = torch.randn(4, 32, 1)
        out = net(x)
        assert out.shape == (4, 32, 1)


class TestGeometricRegularizer:
    def test_losses(self):
        from mnn.quantum_pde.spectral import GeometricPDERegularizer
        reg = GeometricPDERegularizer()
        u = torch.randn(4, 32, 1)
        loss = reg(u)
        assert loss.item() >= 0

    def test_norm_preservation(self):
        from mnn.quantum_pde.spectral import GeometricPDERegularizer
        reg = GeometricPDERegularizer()
        u = torch.randn(4, 32, 1)
        u_ref = torch.randn(4, 32, 1)
        loss = reg.norm_preservation_loss(u, u_ref)
        assert loss.item() >= 0


# ===== Part 7: PDE Discovery =====

class TestPDEDiscovery:
    def test_discover_heat_equation(self):
        from mnn.quantum_pde.pde_discovery import PDEDiscoveryEngine
        # Generate heat equation data: u_t = alpha * u_xx
        alpha = 0.5
        nx, nt = 64, 100
        dx = 2 * np.pi / nx
        dt = 0.001
        x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
        k = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
        u0 = np.sin(x)
        u = np.zeros((nt, nx))
        u_hat = np.fft.fft(u0)
        for t in range(nt):
            u[t] = np.real(np.fft.ifft(u_hat))
            u_hat *= np.exp(-alpha * k**2 * dt)

        engine = PDEDiscoveryEngine(poly_order=2, deriv_order=3, threshold=0.01)
        result = engine.discover(u, dx, dt)
        # Should discover u_xx term with coefficient ~ alpha
        assert "u_xx" in result["active_terms"]
        u_xx_coeff = result["active_coeffs"].get("u_xx", 0)
        assert abs(u_xx_coeff - alpha) < 0.3  # within 0.3 of true value


# ===== Part 8: Categorical PDE =====

class TestCategoricalPDE:
    def test_category_construction(self):
        from mnn.quantum_pde.categorical_pde import PDECategory, SolutionSpace, PDEMorphism
        cat = PDECategory()
        cat.add_space(SolutionSpace("L2", 64, regularity="L2"))
        cat.add_space(SolutionSpace("H1", 64, regularity="H1"))
        cat.add_morphism(PDEMorphism("embed", "L2", "H1"))
        assert len(cat.spaces) == 2
        assert len(cat.morphisms) == 1

    def test_composition(self):
        from mnn.quantum_pde.categorical_pde import PDECategory, SolutionSpace, PDEMorphism
        cat = PDECategory()
        cat.add_space(SolutionSpace("A", 64))
        cat.add_space(SolutionSpace("B", 64))
        cat.add_space(SolutionSpace("C", 64))
        cat.add_morphism(PDEMorphism("f", "A", "B", lambda x: 2*x))
        cat.add_morphism(PDEMorphism("g", "B", "C", lambda x: x+1))
        composed = cat.compose("f", "g")
        assert composed is not None
        result = composed.apply(np.array([1.0, 2.0]))
        assert np.allclose(result, [3.0, 5.0])

    def test_discretization_functor(self):
        from mnn.quantum_pde.categorical_pde import DiscretizationFunctor, SolutionSpace
        functor = DiscretizationFunctor(64, (0, 2*np.pi))
        space = SolutionSpace("H1", 0, regularity="H1")
        disc_space = functor.discretize_space(space)
        assert disc_space.dimension == 64

    def test_discretize_laplacian(self):
        from mnn.quantum_pde.categorical_pde import DiscretizationFunctor
        functor = DiscretizationFunctor(32, (0, 1))
        lap = functor.discretize_laplacian()
        assert lap.shape == (32, 32)
        # Laplacian matrix should be symmetric
        assert np.allclose(lap, lap.T)

    def test_functor_verification(self):
        from mnn.quantum_pde.categorical_pde import DiscretizationFunctor
        functor = DiscretizationFunctor(32, (0, 1))
        lap = functor.discretize_laplacian()
        grad = functor.discretize_gradient()
        test = np.random.randn(32)
        result = functor.verify_functor(lap, grad, test)
        assert result["composition_preserved"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
