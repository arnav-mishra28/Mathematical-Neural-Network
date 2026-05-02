"""
Example 18 — Neural Equation Discovery
========================================
Advanced: Neural SINDy + full chaos diagnostics
"""
import sys; sys.path.insert(0, "..")
import numpy as np

from mnn.advanced.chaos_simulation.simulator import (
    LorenzSimulator, RosslerSimulator, ChenSimulator, SystemFactory
)
from mnn.advanced.chaos_simulation.learner   import DynamicsNet, DynamicsTrainer
from mnn.advanced.chaos_simulation.predictor import ChaosPredictor, EnsemblePredictor
from mnn.advanced.chaos_simulation.discovery import EquationDiscovery, SINDyEngine
from mnn.advanced.chaos_simulation.analyzer  import ChaosNeuralAnalyzer

print("=" * 60)
print("  MNN Example 18 — Neural Equation Discovery")
print("=" * 60)

# ── 1. Compare multiple chaotic systems ───────────────────────
print("\n[1] Simulate three chaotic systems")
lor = LorenzSimulator(); ros = RosslerSimulator(); che = ChenSimulator()
for sim, name in [(lor,"Lorenz"),(ros,"Rössler"),(che,"Chen")]:
    t = sim.simulate(t_span=(0,20), dt=0.01)
    std = t.states.std(axis=0)
    print(f"  {name}: shape={t.states.shape}  std={std.round(2)}")

# ── 2. SINDy on Rössler ────────────────────────────────────────
print("\n[2] SINDy equation discovery on Rössler")
ros_traj = ros.simulate(t_span=(0,30), dt=0.01)
sindy    = SINDyEngine(poly_degree=2, threshold=0.08)
ros_eqs  = sindy.fit(ros_traj, ["x","y","z"])
sindy.print_equations(ros_eqs)

print("  Expected Rössler:  dx/dt = -y-z  dy/dt = x+ay  dz/dt = b+z(x-c)")
for eq in ros_eqs:
    print(f"    {eq.to_string()}  R²={eq.r2_score:.4f}")

# ── 3. Neural SINDy on Lorenz ─────────────────────────────────
print("\n[3] Neural SINDy — learned basis + sparse coefficients")
lor_traj = lor.multi_trajectory(n_traj=3, t_span=(0,10), dt=0.01)
ed       = EquationDiscovery(state_dim=3, library_dim=10, hidden_width=32)
hist     = ed.train_neural_sindy(lor_traj, n_epochs=500, lr=1e-3,
                                   lambda_sparse=0.01, verbose=True, print_every=150)
print(f"  Final recon loss: {hist['recon'][-1]:.5f}")
print(f"  Final sparse loss: {hist['sparse'][-1]:.5f}")
coeffs = ed.get_active_coefficients(["x","y","z"])
print("  Active coefficients:")
for eq_name, terms in coeffs.items():
    print(f"    {eq_name}: {terms}")

# ── 4. Train dynamics net on multiple systems ─────────────────
print("\n[4] Train neural dynamics on Chen system")
che_traj = che.multi_trajectory(n_traj=4, t_span=(0,10), dt=0.005)
che_net  = DynamicsNet(state_dim=3, width=64, depth=4)
che_tr   = DynamicsTrainer(che_net, lr=1e-3)
che_res  = che_tr.train(che_traj, n_epochs=1500, batch_size=256,
                         verbose=True, print_every=500)
err = che_tr.evaluate_derivative_error(che.simulate(t_span=(0,5), dt=0.01))
print(f"  Chen RMSE: {err['rmse']:.5f}")

# ── 5. Chaos diagnostics ──────────────────────────────────────
print("\n[5] Chaos diagnostics on trained Lorenz model")
lor_net  = DynamicsNet(state_dim=3, width=128, depth=5)
lor_traj2= lor.multi_trajectory(n_traj=8, t_span=(0,20), dt=0.01)
lor_tr   = DynamicsTrainer(lor_net, lr=1e-3)
lor_tr.train(lor_traj2, n_epochs=2000, batch_size=512,
             verbose=True, print_every=500)

analyzer = ChaosNeuralAnalyzer(lor_net, state_dim=3, dt=0.01)
x0       = np.array([1.0, 1.0, 1.0])

# Jacobian at fixed points
fps = lor.fixed_points()
for i, fp in enumerate(fps):
    J = analyzer.jacobian_at(fp)
    print(f"  Jacobian at FP{i}: max eigenvalue = {np.linalg.eigvals(J).real.max():.4f}")

# Lyapunov from neural model
lam = analyzer.neural_lyapunov_exponent(x0, n_steps=500, renorm_every=10)
print(f"\n  Neural Lyapunov exponent: λ ≈ {lam:.4f}  (true ≈ 0.9056)")

# Butterfly effect
bf = analyzer.butterfly_effect(x0, epsilon=1e-6, n_steps=300, n_perturbations=3)
print(f"  λ from divergence fit:    {bf['lambda_fit']:.4f}")
print(f"  Error doubling time:      {bf['doubling_time']:.3f} s")

# Ground truth comparison
gt_cmp = analyzer.compare_to_ground_truth(lor, x0, t_end=2.0)
print(f"\n  Ground-truth comparison:")
print(f"    Derivative RMSE:        {gt_cmp['deriv_rmse']:.5f}")
print(f"    Predictability horizon: {gt_cmp['horizon_time']:.3f} s")
print(f"    Attractor scale:        {gt_cmp['attractor_scale']:.4f}")

# Attractor dimension
dim_nn = analyzer.neural_attractor_dimension(x0, n_steps=2000)
print(f"\n  Neural attractor dimension: {dim_nn:.3f}  (Lorenz true ≈ 2.06)")

# ── 6. Ensemble predictor ─────────────────────────────────────
print("\n[6] Ensemble predictor (5 independently trained models)")
models = []
for i in range(5):
    m  = DynamicsNet(state_dim=3, width=64, depth=3)
    tr = DynamicsTrainer(m, lr=1e-3)
    tr.train(lor.simulate(t_span=(0,10),dt=0.01), n_epochs=500, verbose=False)
    models.append(m)

ens = EnsemblePredictor(models, state_dim=3, dt=0.01)
ep  = ens.predict_mean(x0, n_steps=200)
print(f"  Ensemble mean shape: {ep['mean'].shape}")
print(f"  Ensemble std at t=1s: {ep['std'][100].mean():.4f}")

print("\n[OK] Example 18 complete.")
