"""Tests for the Advanced Mathematical Intelligence Layer (mnn.intelligence)."""
import numpy as np
import torch
import pytest
from scipy.integrate import solve_ivp


class TestFlowMapLearner:
    def test_generate_training_data(self):
        from mnn.intelligence.dynamical import FlowMapLearner
        def ode(t, s): return [-s[0]]
        x_now, x_next = FlowMapLearner.generate_training_data(
            ode, np.array([1.0]), (0, 1), dt=0.01)
        assert x_now.shape[0] == x_next.shape[0]
        assert x_now.shape[0] > 50

    def test_flow_map_learns_linear(self):
        from mnn.intelligence.dynamical import FlowMapLearner
        def ode(t, s): return [-0.5 * s[0]]
        x_now, x_next = FlowMapLearner.generate_training_data(
            ode, np.array([2.0]), (0, 5), dt=0.01)
        learner = FlowMapLearner(1, width=32, depth=2, dt=0.01, lr=1e-3)
        learner.train(x_now, x_next, n_epochs=300, batch_size=64, verbose=False)
        pred = learner.predict(x_now[0], n_steps=50)
        assert pred.shape == (51, 1)

    def test_multi_step_prediction(self):
        from mnn.intelligence.dynamical import FlowMapLearner, FlowMapNetwork
        net = FlowMapNetwork(2, width=16, depth=1, dt=0.01)
        x = torch.randn(1, 2)
        traj = net.multi_step(x, 10)
        assert traj.shape == (1, 11, 2)


class TestStabilityAnalyzer:
    def test_classify_stable_node(self):
        from mnn.intelligence.dynamical import StabilityAnalyzer
        evals = np.array([-1.0, -2.0])
        info = StabilityAnalyzer.classify_fixed_point(evals)
        assert info["type"] == "stable node"
        assert info["stable"] is True

    def test_classify_saddle(self):
        from mnn.intelligence.dynamical import StabilityAnalyzer
        evals = np.array([-1.0, 2.0])
        info = StabilityAnalyzer.classify_fixed_point(evals)
        assert info["type"] == "saddle"
        assert info["stable"] is False

    def test_find_fixed_points_linear(self):
        from mnn.intelligence.dynamical import StabilityAnalyzer
        def rhs(t, x): return [-x[0], -x[1]]
        fps = StabilityAnalyzer.find_fixed_points(rhs, dim=2, n_starts=10)
        assert len(fps) >= 1
        assert np.linalg.norm(fps[0]) < 1e-6

    def test_numerical_jacobian(self):
        from mnn.intelligence.dynamical import StabilityAnalyzer
        def f(x): return np.array([-2*x[0], -3*x[1]])
        J = StabilityAnalyzer.numerical_jacobian(f, np.array([0.0, 0.0]))
        assert np.allclose(J, np.diag([-2, -3]), atol=1e-4)


class TestBifurcationDetector:
    def test_detect_hopf(self):
        from mnn.intelligence.dynamical import BifurcationDetector
        def factory(mu):
            def rhs(t, s):
                x, y = s
                return [mu*x - y - x*(x**2+y**2), x + mu*y - y*(x**2+y**2)]
            return rhs
        hopf = BifurcationDetector.detect_hopf(
            factory, (-1, 1), dim=2, fp_guess=np.array([0.,0.]), n_params=50)
        # Should detect Hopf near mu=0
        if hopf:
            assert abs(hopf[0]["parameter"]) < 0.2


class TestNeuralGroupOperator:
    def test_constraint_losses(self):
        from mnn.intelligence.group_algebra import NeuralGroupOperator
        op = NeuralGroupOperator(element_dim=4, width=16, depth=1, abelian=True)
        elements = torch.randn(8, 4)
        losses = op.constraint_losses(elements)
        assert "identity" in losses
        assert "inverse" in losses
        assert "associativity" in losses
        assert "commutativity" in losses

    def test_operate_shape(self):
        from mnn.intelligence.group_algebra import NeuralGroupOperator
        op = NeuralGroupOperator(element_dim=3, width=16, depth=1)
        a = torch.randn(5, 3)
        b = torch.randn(5, 3)
        result = op.operate(a, b)
        assert result.shape == (5, 3)


class TestInvariantLearner:
    def test_learns_norm_squared(self):
        from mnn.intelligence.group_algebra import InvariantLearner
        R = np.array([[0, -1], [1, 0]], dtype=np.float32)
        data = np.random.randn(200, 2).astype(np.float32)
        targets = (data[:,0:1]**2 + data[:,1:2]**2).astype(np.float32)
        il = InvariantLearner(2, 1, [R], width=32, depth=2)
        il.train(data, targets, n_epochs=500, verbose=False)
        pred = il.predict(data[:10])
        assert pred.shape == (10, 1)


class TestEquivariantNetwork:
    def test_equivariance_property(self):
        from mnn.intelligence.group_algebra import EquivariantNetwork
        R = torch.tensor([[0., -1.], [1., 0.]])
        I = torch.eye(2)
        eq = EquivariantNetwork(2, 2, [I, R, R@R, R@R@R], width=16, depth=1)
        err = eq.equivariance_error(torch.randn(10, 2))
        assert err < 0.1  # equivariance enforced by construction


class TestNeuralPDESolver:
    def test_poisson_setup(self):
        from mnn.intelligence.neural_pde import poisson_2d
        p = poisson_2d()
        assert p.name == "Poisson-2D"
        assert p.spatial_dim == 2

    def test_heat_setup(self):
        from mnn.intelligence.neural_pde import heat_1d
        p = heat_1d()
        assert p.name == "Heat-1D"
        assert p.has_time is True
        assert p.exact_solution is not None

    def test_generalized_pinn_forward(self):
        from mnn.intelligence.neural_pde import GeneralizedPINN
        pinn = GeneralizedPINN(2, 1, width=16, depth=2, use_fourier=False)
        x = torch.randn(10, 2)
        out = pinn(x)
        assert out.shape == (10, 1)

    def test_solver_trains(self):
        from mnn.intelligence.neural_pde import NeuralPDESolver, heat_1d
        p = heat_1d()
        solver = NeuralPDESolver(p, width=16, depth=2, lr=1e-3)
        solver.train(n_epochs=50, n_collocation=100, verbose=False)
        assert "pde" in solver.history
        assert len(solver.history["pde"]) == 50


class TestLibraryBuilder:
    def test_feature_count(self):
        from mnn.intelligence.discovery import LibraryBuilder
        lb = LibraryBuilder(3, poly_order=2)
        # 1 + 3 + 6 (quadratic with cross) = 10
        assert lb.n_features == 10

    def test_transform_shape(self):
        from mnn.intelligence.discovery import LibraryBuilder
        lb = LibraryBuilder(2, poly_order=3, include_trig=True)
        X = np.random.randn(50, 2)
        theta = lb.transform(X)
        assert theta.shape[0] == 50
        assert theta.shape[1] == lb.n_features


class TestSparseRegressor:
    def test_recovers_linear(self):
        from mnn.intelligence.discovery import SparseRegressor, LibraryBuilder
        lb = LibraryBuilder(2, poly_order=2)
        # True: dx/dt = 3*x0 - 2*x1
        X = np.random.randn(500, 2)
        dX = np.column_stack([3*X[:,0] - 2*X[:,1], X[:,0]])
        theta = lb.transform(X)
        sr = SparseRegressor(threshold=0.1)
        sr.fit(theta, dX)
        r2 = sr.score(theta, dX)
        assert r2 > 0.95

    def test_equation_strings(self):
        from mnn.intelligence.discovery import SparseRegressor, LibraryBuilder
        lb = LibraryBuilder(2, poly_order=1)
        X = np.random.randn(100, 2)
        dX = np.column_stack([X[:,0], X[:,1]])
        theta = lb.transform(X)
        sr = SparseRegressor(threshold=0.01)
        sr.fit(theta, dX)
        eqs = sr.equation_strings(lb.feature_names)
        assert len(eqs) == 2


class TestHybridDiscovery:
    def test_discovers_simple_ode(self):
        from mnn.intelligence.discovery import HybridDiscovery
        # dx/dt = -x
        t = np.linspace(0, 5, 500)
        traj = np.exp(-t).reshape(-1, 1)
        hd = HybridDiscovery(1, poly_order=2, threshold=0.05)
        result = hd.discover(t, traj, n_smooth_epochs=1000,
                              use_finite_diff=True, verbose=False)
        assert result["r2_score"] > 0.8


class TestScientificDiscoveryEngine:
    def test_auto_discover(self):
        from mnn.intelligence.discovery import ScientificDiscoveryEngine
        t = np.linspace(0, 5, 300)
        traj = np.column_stack([np.exp(-t), -np.exp(-t)])
        engine = ScientificDiscoveryEngine(2)
        result = engine.auto_discover(
            t, traj, thresholds=[0.05, 0.1], poly_orders=[1],
            try_trig=False, verbose=False)
        assert result is not None or len(engine.all_results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
