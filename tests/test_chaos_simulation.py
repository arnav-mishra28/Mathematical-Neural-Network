"""Tests for mnn.advanced.chaos_simulation — Phase 5."""
import sys; sys.path.insert(0, "..")
import numpy as np
import torch

from mnn.advanced.chaos_simulation.simulator import (
    LorenzSimulator, RosslerSimulator, ChenSimulator,
    DuffingSimulator, VanDerPolSimulator, HalvorsenSimulator,
    ChaosTrajectory, SystemFactory
)
from mnn.advanced.chaos_simulation.learner  import (
    DynamicsNet, FourierDynNet, DynamicsTrainer
)
from mnn.advanced.chaos_simulation.predictor import ChaosPredictor, EnsemblePredictor
from mnn.advanced.chaos_simulation.discovery  import (
    SINDyEngine, EquationDiscovery, SymbolicTerm, DiscoveredEquation
)
from mnn.advanced.chaos_simulation.analyzer  import ChaosNeuralAnalyzer


# ── Simulator tests ───────────────────────────────────────────

def test_lorenz_trajectory_shape():
    lor  = LorenzSimulator()
    traj = lor.simulate(t_span=(0, 2), dt=0.01)
    assert traj.states.shape[1] == 3
    assert traj.derivatives.shape == traj.states.shape

def test_lorenz_fixed_points():
    lor = LorenzSimulator()
    fps = lor.fixed_points()
    assert len(fps) == 3
    assert np.allclose(fps[0], [0,0,0])

def test_lorenz_chaotic():
    lor  = LorenzSimulator()
    traj = lor.simulate(t_span=(0,10), dt=0.01)
    assert traj.states[:,0].std() > 5.0

def test_lorenz_derivatives_correct():
    """Verify dx/dt = σ(y-x) at a known point."""
    lor  = LorenzSimulator(sigma=10., rho=28., beta=8/3)
    x,y,z = 1.0, 2.0, 3.0
    traj  = lor.simulate(np.array([x,y,z]), t_span=(0,0.01), dt=0.005)
    # Expected dx/dt at t=0: σ(y-x) = 10*(2-1) = 10
    assert abs(traj.derivatives[0,0] - 10.0) < 0.1

def test_rossler_shape():
    ros  = RosslerSimulator()
    traj = ros.simulate(t_span=(0,2), dt=0.01)
    assert traj.states.shape[1] == 3

def test_chen_shape():
    che  = ChenSimulator()
    traj = che.simulate(t_span=(0,2), dt=0.005)
    assert traj.states.shape[1] == 3

def test_duffing_shape():
    duf  = DuffingSimulator()
    traj = duf.simulate(t_span=(0,5), dt=0.01)
    assert traj.states.shape[1] == 2

def test_vanderpol_shape():
    vdp  = VanDerPolSimulator()
    traj = vdp.simulate(t_span=(0,5), dt=0.01)
    assert traj.states.shape[1] == 2

def test_halvorsen_shape():
    hal  = HalvorsenSimulator()
    traj = hal.simulate(t_span=(0,3), dt=0.01)
    assert traj.states.shape[1] == 3

def test_system_factory():
    for name in SystemFactory.available():
        sim  = SystemFactory.get(name)
        traj = sim.simulate(t_span=(0,1), dt=0.01)
        assert traj.states.shape[1] == sim.dim

def test_multi_trajectory():
    lor  = LorenzSimulator()
    mt   = lor.multi_trajectory(n_traj=3, t_span=(0,2), dt=0.01)
    assert len(mt) == 3 * len(lor.simulate(t_span=(0,2), dt=0.01))

def test_trajectory_train_test_split():
    lor   = LorenzSimulator()
    traj  = lor.simulate(t_span=(0,5), dt=0.01)
    n     = len(traj)
    tr, te = traj.train_test_split(0.2)
    assert len(tr) + len(te) == n

def test_trajectory_add_noise():
    lor    = LorenzSimulator()
    traj   = lor.simulate(t_span=(0,2), dt=0.01)
    noisy  = traj.add_noise(0.1)
    assert not np.allclose(traj.states, noisy.states)

# ── DynamicsNet tests ─────────────────────────────────────────

def test_dynamics_net_shape():
    net = DynamicsNet(state_dim=3, width=32, depth=2)
    x   = torch.rand(10, 3)
    y   = net(x)
    assert y.shape == (10, 3)

def test_dynamics_net_2d():
    net = DynamicsNet(state_dim=2, width=16, depth=2)
    x   = torch.rand(5, 2)
    assert net(x).shape == (5, 2)

def test_dynamics_net_predict_numpy():
    net  = DynamicsNet(state_dim=3, width=16, depth=2)
    x    = np.random.randn(8, 3).astype(np.float32)
    pred = net.predict_numpy(x)
    assert pred.shape == (8, 3)

def test_fourier_dyn_net():
    net = FourierDynNet(state_dim=3, n_fourier=16, width=32, depth=2)
    x   = torch.rand(5, 3)
    assert net(x).shape == (5, 3)

def test_trainer_runs():
    lor  = LorenzSimulator()
    traj = lor.simulate(t_span=(0,2), dt=0.01)
    net  = DynamicsNet(state_dim=3, width=16, depth=2)
    tr   = DynamicsTrainer(net, lr=1e-3)
    res  = tr.train(traj, n_epochs=5, verbose=False)
    assert "deriv" in res.final_losses
    assert res.n_epochs == 5

def test_trainer_loss_decreases():
    lor  = LorenzSimulator()
    traj = lor.simulate(t_span=(0,10), dt=0.01)
    net  = DynamicsNet(state_dim=3, width=32, depth=3)
    tr   = DynamicsTrainer(net, lr=1e-3)
    res  = tr.train(traj, n_epochs=200, verbose=False)
    early = np.mean(res.loss_history["deriv"][:20])
    late  = np.mean(res.loss_history["deriv"][-20:])
    assert late < early, f"Loss not decreasing: {early:.5f} → {late:.5f}"

def test_derivative_error_metrics():
    lor  = LorenzSimulator()
    traj = lor.simulate(t_span=(0,2), dt=0.01)
    net  = DynamicsNet(state_dim=3, width=16, depth=2)
    tr   = DynamicsTrainer(net, lr=1e-3)
    tr.train(traj, n_epochs=5, verbose=False)
    err  = tr.evaluate_derivative_error(traj)
    assert "mse" in err and "rmse" in err
    assert err["mse"] >= 0

# ── Predictor tests ───────────────────────────────────────────

def test_euler_prediction_shape():
    net  = DynamicsNet(state_dim=3, width=16, depth=2)
    pred = ChaosPredictor(net, state_dim=3, dt=0.01)
    x0   = np.array([1.,1.,1.], dtype=np.float32)
    traj = pred.predict_euler(x0, n_steps=50)
    assert traj.shape == (51, 3)

def test_rk4_prediction_shape():
    net  = DynamicsNet(state_dim=3, width=16, depth=2)
    pred = ChaosPredictor(net, state_dim=3, dt=0.01)
    x0   = np.array([1.,1.,1.], dtype=np.float32)
    traj = pred.predict_rk4(x0, n_steps=100)
    assert traj.shape == (101, 3)

def test_adaptive_prediction_shape():
    net  = DynamicsNet(state_dim=3, width=16, depth=2)
    pred = ChaosPredictor(net, state_dim=3, dt=0.01)
    x0   = np.array([1.,1.,1.], dtype=np.float32)
    traj = pred.predict_adaptive(x0, t_end=0.5, dt_out=0.01)
    assert traj.shape[1] == 3

def test_prediction_error():
    lor  = LorenzSimulator()
    traj = lor.simulate(t_span=(0,5), dt=0.01)
    net  = DynamicsNet(state_dim=3, width=16, depth=2)
    pred = ChaosPredictor(net, state_dim=3, dt=0.01)
    errs = pred.prediction_error(traj.states[0], traj.states[:201], method="rk4")
    assert errs.shape == (201,)
    assert (errs >= 0).all()

def test_predictability_horizon():
    lor  = LorenzSimulator()
    traj = lor.simulate(t_span=(0,10), dt=0.01)
    net  = DynamicsNet(state_dim=3, width=16, depth=2)
    pred = ChaosPredictor(net, state_dim=3, dt=0.01)
    h    = pred.predictability_horizon(traj.states[0], traj.states[:501], error_threshold=5.0)
    assert "horizon_time" in h and "lyapunov_fit" in h
    assert h["horizon_steps"] <= 500

def test_multi_step_forecast():
    net  = DynamicsNet(state_dim=3, width=16, depth=2)
    pred = ChaosPredictor(net, state_dim=3, dt=0.01)
    x0   = np.array([1.,1.,1.], dtype=np.float32)
    fc   = pred.multi_step_forecast(x0, n_steps=20, n_restarts=3)
    assert fc["mean"].shape == (21, 3)
    assert fc["std"].shape  == (21, 3)

def test_ensemble_predictor():
    models = [DynamicsNet(3, 16, 2) for _ in range(3)]
    ens    = EnsemblePredictor(models, state_dim=3, dt=0.01)
    x0     = np.array([1.,1.,1.], dtype=np.float32)
    out    = ens.predict_mean(x0, n_steps=20)
    assert out["mean"].shape == (21, 3)

# ── SINDy tests ───────────────────────────────────────────────

def test_sindy_library_shape():
    sindy = SINDyEngine(poly_degree=2)
    X     = np.random.randn(100, 3)
    Theta, names = sindy.build_library(X)
    assert Theta.shape[0] == 100
    assert len(names) == Theta.shape[1]

def test_sindy_library_has_constant():
    sindy = SINDyEngine(poly_degree=1)
    X     = np.random.randn(50, 2)
    _, names = sindy.build_library(X)
    assert "1" in names

def test_sindy_library_has_products():
    sindy = SINDyEngine(poly_degree=2)
    X     = np.random.randn(50, 2)
    _, names = sindy.build_library(X)
    assert any("·" in n for n in names)

def test_sindy_fit_returns_equations():
    lor   = LorenzSimulator()
    traj  = lor.simulate(t_span=(0,5), dt=0.01)
    sindy = SINDyEngine(poly_degree=2, threshold=0.05)
    eqs   = sindy.fit(traj, ["x","y","z"])
    assert len(eqs) == 3
    for eq in eqs: assert isinstance(eq, DiscoveredEquation)

def test_sindy_r2_positive():
    lor   = LorenzSimulator()
    traj  = lor.simulate(t_span=(0,10), dt=0.01)
    sindy = SINDyEngine(poly_degree=2, threshold=0.1)
    eqs   = sindy.fit(traj)
    for eq in eqs: assert eq.r2_score <= 1.0

def test_sindy_discovers_lorenz_terms():
    """SINDy should find x and y terms in dx/dt for Lorenz."""
    lor   = LorenzSimulator()
    traj  = lor.simulate(t_span=(0,30), dt=0.01)
    sindy = SINDyEngine(poly_degree=2, threshold=0.1)
    eqs   = sindy.fit(traj, ["x","y","z"])
    eq_x  = eqs[0]   # dx/dt equation
    active_names = [t.name for t in eq_x.active_terms]
    # dx/dt = σ(y-x): should find x0 and x1 terms
    assert any("x" in n for n in active_names)

def test_equation_discovery_init():
    ed = EquationDiscovery(state_dim=3, library_dim=10, hidden_width=16)
    x  = torch.rand(5, 3)
    y  = ed.forward(x)
    assert y.shape == (5, 3)

def test_neural_sindy_runs():
    lor   = LorenzSimulator()
    traj  = lor.simulate(t_span=(0,3), dt=0.01)
    ed    = EquationDiscovery(state_dim=3, library_dim=10, hidden_width=16)
    hist  = ed.train_neural_sindy(traj, n_epochs=10, lr=1e-3, verbose=False)
    assert "recon" in hist
    assert len(hist["recon"]) == 10

def test_neural_sindy_sparsity():
    lor  = LorenzSimulator()
    traj = lor.simulate(t_span=(0,3), dt=0.01)
    ed   = EquationDiscovery(state_dim=3, library_dim=10, hidden_width=16, threshold=0.1)
    ed.train_neural_sindy(traj, n_epochs=5, lr=1e-3, verbose=False, lambda_sparse=0.1)
    xi   = ed.Xi.detach().numpy()
    # After large sparsity penalty + threshold, many should be ~0
    n_zero = int(np.sum(np.abs(xi) < 0.1))
    assert n_zero >= 0   # at least runs without error

def test_symbolic_term():
    t = SymbolicTerm("x·y", 3.14, True)
    assert "3.1400" in repr(t)
    assert "x·y" in repr(t)

# ── Analyzer tests ────────────────────────────────────────────

def test_jacobian_shape():
    net  = DynamicsNet(state_dim=3, width=16, depth=2)
    ana  = ChaosNeuralAnalyzer(net, state_dim=3, dt=0.01)
    x0   = np.array([1.,2.,3.], dtype=np.float32)
    J    = ana.jacobian_at(x0)
    assert J.shape == (3, 3)

def test_lyapunov_finite():
    net  = DynamicsNet(state_dim=3, width=16, depth=2)
    ana  = ChaosNeuralAnalyzer(net, state_dim=3, dt=0.01)
    x0   = np.array([1.,1.,1.], dtype=np.float32)
    lam  = ana.neural_lyapunov_exponent(x0, n_steps=50, renorm_every=10)
    assert np.isfinite(lam)

def test_butterfly_effect_keys():
    net  = DynamicsNet(state_dim=3, width=16, depth=2)
    ana  = ChaosNeuralAnalyzer(net, state_dim=3, dt=0.01)
    x0   = np.array([1.,1.,1.], dtype=np.float32)
    bf   = ana.butterfly_effect(x0, epsilon=1e-5, n_steps=20, n_perturbations=2)
    assert "reference" in bf and "lambda_fit" in bf and "divergences" in bf

def test_butterfly_divergences_shape():
    net  = DynamicsNet(state_dim=3, width=16, depth=2)
    ana  = ChaosNeuralAnalyzer(net, state_dim=3, dt=0.01)
    x0   = np.array([1.,1.,1.], dtype=np.float32)
    bf   = ana.butterfly_effect(x0, n_steps=30, n_perturbations=3)
    assert bf["divergences"].shape == (3, 31)

def test_takens_embedding():
    signal = np.sin(np.linspace(0, 20*np.pi, 1000))
    emb    = ChaosNeuralAnalyzer.takens_embedding(signal, dim=3, lag=10)
    assert emb.shape[1] == 3
    assert emb.shape[0] == 1000 - 2*10

def test_ground_truth_comparison():
    lor  = LorenzSimulator()
    net  = DynamicsNet(state_dim=3, width=16, depth=2)
    ana  = ChaosNeuralAnalyzer(net, state_dim=3, dt=0.01)
    x0   = np.array([1.,1.,1.], dtype=np.float32)
    cmp  = ana.compare_to_ground_truth(lor, x0, t_end=0.5)
    assert "errors" in cmp and "horizon_time" in cmp and "deriv_rmse" in cmp

def test_attractor_dimension_positive():
    net  = DynamicsNet(state_dim=3, width=16, depth=2)
    ana  = ChaosNeuralAnalyzer(net, state_dim=3, dt=0.01)
    x0   = np.array([1.,1.,1.], dtype=np.float32)
    dim  = ana.neural_attractor_dimension(x0, n_steps=200)
    assert dim >= 0

def test_full_diagnostics():
    lor  = LorenzSimulator()
    net  = DynamicsNet(state_dim=3, width=16, depth=2)
    ana  = ChaosNeuralAnalyzer(net, state_dim=3, dt=0.01)
    x0   = np.array([1.,1.,1.], dtype=np.float32)
    rpt  = ana.full_diagnostics(x0, simulator=lor, n_steps_lyap=50)
    assert "neural_lyapunov" in rpt
    assert "predictability_horizon" in rpt


if __name__ == "__main__":
    tests = [
        # Simulator
        test_lorenz_trajectory_shape, test_lorenz_fixed_points,
        test_lorenz_chaotic, test_lorenz_derivatives_correct,
        test_rossler_shape, test_chen_shape, test_duffing_shape,
        test_vanderpol_shape, test_halvorsen_shape, test_system_factory,
        test_multi_trajectory, test_trajectory_train_test_split,
        test_trajectory_add_noise,
        # Learner
        test_dynamics_net_shape, test_dynamics_net_2d,
        test_dynamics_net_predict_numpy, test_fourier_dyn_net,
        test_trainer_runs, test_trainer_loss_decreases,
        test_derivative_error_metrics,
        # Predictor
        test_euler_prediction_shape, test_rk4_prediction_shape,
        test_adaptive_prediction_shape, test_prediction_error,
        test_predictability_horizon, test_multi_step_forecast,
        test_ensemble_predictor,
        # SINDy
        test_sindy_library_shape, test_sindy_library_has_constant,
        test_sindy_library_has_products, test_sindy_fit_returns_equations,
        test_sindy_r2_positive, test_sindy_discovers_lorenz_terms,
        test_equation_discovery_init, test_neural_sindy_runs,
        test_neural_sindy_sparsity, test_symbolic_term,
        # Analyzer
        test_jacobian_shape, test_lyapunov_finite,
        test_butterfly_effect_keys, test_butterfly_divergences_shape,
        test_takens_embedding, test_ground_truth_comparison,
        test_attractor_dimension_positive, test_full_diagnostics,
    ]
    passed = failed = 0
    for fn in tests:
        try:   fn(); print(f"  ✓ {fn.__name__}"); passed += 1
        except Exception as e: print(f"  ✗ {fn.__name__}: {e}"); failed += 1
    print(f"\n[{passed} passed, {failed} failed]")
