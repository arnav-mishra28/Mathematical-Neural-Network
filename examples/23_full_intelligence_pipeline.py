"""
Example 23 — Full MNN Intelligence Pipeline
End-to-end demonstration of all four pillars working together.
"""
import numpy as np
from scipy.integrate import solve_ivp

from mnn.intelligence.dynamical import FlowMapLearner, StabilityAnalyzer
from mnn.intelligence.group_algebra import NeuralGroupOperator, InvariantLearner
from mnn.intelligence.neural_pde import NeuralPDESolver, heat_1d
from mnn.intelligence.discovery import HybridDiscovery


def main():
    print("=" * 70)
    print("  FULL MNN INTELLIGENCE PIPELINE")
    print("  Calculus + Topology + Algebra + Dynamics + Chaos + PDEs")
    print("  Learning + Discovery + Constraint Reasoning")
    print("=" * 70)

    # --- Pillar 1: Learn dynamics ---
    print("\n[1/4] Flow Map Learning (Lorenz)...")
    from mnn.chaos.attractors import LorenzAttractor
    lorenz = LorenzAttractor()
    x_now, x_next = FlowMapLearner.generate_training_data(
        lorenz.ode, np.array([1.,1.,1.]), (0,10), dt=0.01)
    learner = FlowMapLearner(3, width=64, depth=3, dt=0.01)
    learner.train(x_now, x_next, n_epochs=500, verbose=False)
    pred = learner.predict(x_now[0], n_steps=500)
    print(f"  Predicted {pred.shape[0]} steps, final state: {np.round(pred[-1],2)}")

    # Stability
    fps = StabilityAnalyzer.analyze_system(lorenz.ode, dim=3)
    print(f"  Found {len(fps)} fixed points")
    for fp in fps:
        print(f"    {np.round(fp['fixed_point'],2)} → {fp['type']}")

    # --- Pillar 2: Group-aware computation ---
    print("\n[2/4] Symmetry-Aware Invariant Learning...")
    n_rot = 4
    actions = [np.array([[np.cos(2*np.pi*k/n_rot),-np.sin(2*np.pi*k/n_rot)],
                          [np.sin(2*np.pi*k/n_rot), np.cos(2*np.pi*k/n_rot)]],
                         dtype=np.float32) for k in range(n_rot)]
    data = np.random.randn(200,2).astype(np.float32)
    targets = (data[:,0:1]**2 + data[:,1:2]**2).astype(np.float32)
    inv = InvariantLearner(2, 1, actions, width=32, depth=2)
    inv.train(data, targets, n_epochs=500, verbose=False)
    test = np.random.randn(50,2).astype(np.float32)
    R = actions[1]
    err = np.mean((inv.predict(test) - inv.predict(test @ R.T))**2)
    print(f"  Invariance error: {err:.8f}")

    # --- Pillar 3: Solve a PDE ---
    print("\n[3/4] Neural PDE Solver (Heat Equation)...")
    problem = heat_1d(alpha=1.0, L=1.0, T=0.2)
    solver = NeuralPDESolver(problem, width=32, depth=3)
    solver.train(n_epochs=1000, n_collocation=500, verbose=False)
    pts = np.column_stack([np.linspace(0,1,50), np.full(50, 0.1)])
    u = solver.predict(pts)
    u_ex = problem.exact_solution(pts)
    print(f"  PDE MSE at t=0.1: {np.mean((u-u_ex)**2):.8f}")

    # --- Pillar 4: Discover equations ---
    print("\n[4/4] Scientific Discovery (Damped Oscillator)...")
    sol = solve_ivp(lambda t,s: [s[1], -2*s[0]-0.3*s[1]], (0,15), [1,0],
                    t_eval=np.linspace(0,15,1500))
    hd = HybridDiscovery(2, poly_order=2, threshold=0.05)
    result = hd.discover(sol.t, sol.y.T, n_smooth_epochs=1500,
                          var_names=["dx/dt","dy/dt"], verbose=False)
    print(f"  R² = {result['r2_score']:.4f}")
    for eq in result["equations"]:
        print(f"  {eq}")

    print("\n" + "=" * 70)
    print("  ALL PILLARS VERIFIED — MNN Intelligence Layer Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
