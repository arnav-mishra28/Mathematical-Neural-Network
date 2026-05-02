"""
Example 17 — Lorenz Chaos Engine
==================================
PART 1: Simulate the Lorenz system
PART 2: Train MNN to learn dx/dt = f(x)
PART 3: Predict trajectories + demonstrate butterfly effect
PART 4: Discover equations via SINDy
"""
import sys; sys.path.insert(0, "..")
import numpy as np

from mnn.advanced.chaos_simulation import (
    ChaosSimulator, LorenzSimulator,
    NeuralDynamicsLearner, DynamicsTrainer, DynamicsResult,
    ChaosPredictor, EquationDiscovery
)
from mnn.advanced.chaos_simulation.learner import DynamicsNet
from mnn.advanced.chaos_simulation.analyzer import ChaosNeuralAnalyzer
from mnn.advanced.chaos_simulation.discovery import SINDyEngine

print("=" * 60)
print("  MNN Example 17 — Lorenz Chaos Engine")
print("=" * 60)

# ── PART 1: Simulate Lorenz ───────────────────────────────────
print("\n[PART 1] Simulate Lorenz system")
lor = LorenzSimulator(sigma=10.0, rho=28.0, beta=8/3)
print(f"  {lor}")
print(f"  Fixed points: {len(lor.fixed_points())}")

traj = lor.simulate(t_span=(0, 30), dt=0.01)
print(f"  {traj}")
print(f"  X range: [{traj.states[:,0].min():.2f}, {traj.states[:,0].max():.2f}]")
print(f"  Theoretical λ₁ ≈ {lor.theoretical_lyapunov()}")

# Multi-trajectory for richer training data
multi = lor.multi_trajectory(n_traj=5, t_span=(0,15), dt=0.01)
print(f"  Multi-traj: {multi}")

# ── PART 2: Train MNN on Lorenz dynamics ──────────────────────
print("\n[PART 2] Train Neural Dynamics: state → dx/dt")
net     = DynamicsNet(state_dim=3, width=128, depth=5)
trainer = DynamicsTrainer(net, lr=1e-3)
print(f"  {net}")

result = trainer.train(
    multi, n_epochs=3000, batch_size=512,
    verbose=True, print_every=600,
    w_deriv=1.0
)
print(f"\n{result.summary()}")

# Evaluate derivative accuracy
err = trainer.evaluate_derivative_error(traj)
print(f"\n  Derivative errors:")
for k,v in err.items(): print(f"    {k}: {v:.6f}")

# ── PART 3: Predict trajectories ──────────────────────────────
print("\n[PART 3] Trajectory prediction")
predictor = ChaosPredictor(net, state_dim=3, dt=0.01)
x0 = traj.states[0]

# Euler vs RK4
pred_euler = predictor.predict_euler(x0, n_steps=200)
pred_rk4   = predictor.predict_rk4(x0, n_steps=500)
print(f"  Euler prediction shape:  {pred_euler.shape}")
print(f"  RK4   prediction shape:  {pred_rk4.shape}")

# Predictability horizon
horizon = predictor.predictability_horizon(
    x0, traj.states[:501], error_threshold=2.0
)
print(f"\n  Predictability horizon:")
print(f"    Steps:          {horizon['horizon_steps']}")
print(f"    Time:           {horizon['horizon_time']:.3f} s")
print(f"    λ (fit):        {horizon['lyapunov_fit']:.4f}")

# Butterfly effect demo
print("\n  Butterfly effect (ε = 1e-6 perturbation):")
ensemble = predictor.multi_step_forecast(x0, n_steps=300, n_restarts=5, noise_std=1e-6)
print(f"    Mean traj shape: {ensemble['mean'].shape}")
print(f"    Spread at t=1s:  {ensemble['std'][100].mean():.2e}")
print(f"    Spread at t=3s:  {ensemble['std'][300].mean():.2e}")

# ── PART 4: Equation Discovery via SINDy ──────────────────────
print("\n[PART 4] SINDy Equation Discovery")
sindy  = SINDyEngine(poly_degree=2, threshold=0.05, max_iter=10)
eqs    = sindy.fit(traj, var_names=["x","y","z"])
sindy.print_equations(eqs)

print("\n  Expected Lorenz equations:")
print("    dx/dt = σ(y-x)  = -10x + 10y")
print("    dy/dt = ρx - y - xz  = 28x - y - xz")
print("    dz/dt = xy - βz = xy - 2.667z")

print("\n  Discovered active terms:")
for eq in eqs:
    print(f"    {eq.to_string()}")
    print(f"    R² = {eq.r2_score:.4f}")

print("\n[OK] Example 17 complete.")
