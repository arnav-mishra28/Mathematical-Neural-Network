"""
Example 19 — Advanced Mathematical Intelligence: Dynamical Systems
Demonstrates flow-map learning on the Lorenz attractor.
"""
import numpy as np
import matplotlib.pyplot as plt

from mnn.intelligence.dynamical import FlowMapLearner, StabilityAnalyzer, BifurcationDetector
from mnn.chaos.attractors import LorenzAttractor


def main():
    print("=" * 60)
    print("  PILLAR 1: Dynamical + Nonlinear Systems")
    print("=" * 60)

    # --- 1a. Flow Map Learning on Lorenz ---
    print("\n[1a] Learning Lorenz flow map x(t+Δt) = F(x(t))...")
    lorenz = LorenzAttractor(sigma=10, rho=28, beta=8/3)
    dt = 0.01

    x_now, x_next = FlowMapLearner.generate_training_data(
        lorenz.ode, x0=np.array([1., 1., 1.]),
        t_span=(0, 30), dt=dt
    )
    print(f"  Training data: {x_now.shape[0]} pairs")

    learner = FlowMapLearner(state_dim=3, width=128, depth=4, dt=dt)
    learner.train(x_now, x_next, n_epochs=1000, batch_size=256,
                  multi_step_weight=0.1, print_every=200)

    # Predict trajectory
    pred = learner.predict(x_now[0], n_steps=2000)
    print(f"  Predicted trajectory shape: {pred.shape}")

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot(x_now[:2000, 0], x_now[:2000, 1], x_now[:2000, 2], alpha=0.5, lw=0.5, label="True")
    ax1.plot(pred[:, 0], pred[:, 1], pred[:, 2], alpha=0.5, lw=0.5, label="Learned")
    ax1.set_title("Lorenz: True vs Learned Flow Map")
    ax1.legend()

    # --- 1b. Stability Analysis ---
    print("\n[1b] Stability analysis of Lorenz system...")
    results = StabilityAnalyzer.analyze_system(lorenz.ode, dim=3, search_range=30)
    for r in results:
        print(f"  Fixed point: {np.round(r['fixed_point'], 3)}")
        print(f"    Type: {r['type']}")
        print(f"    Eigenvalues: {np.round(r['eigenvalues'], 3)}")

    # --- 1c. Bifurcation Detection ---
    print("\n[1c] Bifurcation detection (Lorenz ρ sweep)...")

    def lorenz_factory(rho):
        def rhs(t, s):
            x, y, z = s
            return [10 * (y - x), x * (rho - z) - y, x * y - (8/3) * z]
        return rhs

    hopf_pts = BifurcationDetector.detect_hopf(
        lorenz_factory, param_range=(0, 30), dim=3,
        fp_guess=np.array([0., 0., 0.]), n_params=100
    )
    for h in hopf_pts:
        print(f"  Hopf bifurcation at ρ ≈ {h['parameter']:.2f}, freq={h['frequency']:.3f}")

    # Loss curve
    ax2 = fig.add_subplot(132)
    ax2.semilogy(learner.history["loss"])
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.set_title("Flow Map Training Loss")

    # Per-component comparison
    ax3 = fig.add_subplot(133)
    t_true = np.arange(2000) * dt
    for i, label in enumerate(["x", "y", "z"]):
        ax3.plot(t_true, x_now[:2000, i], "-", alpha=0.6, label=f"True {label}")
        ax3.plot(t_true[:len(pred)], pred[:min(2000, len(pred)), i], "--", alpha=0.6, label=f"Pred {label}")
    ax3.set_xlabel("Time"); ax3.set_ylabel("State"); ax3.set_title("Component Comparison")
    ax3.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("dynamical_systems_demo.png", dpi=150)
    plt.show()
    print("\nDone! Saved: dynamical_systems_demo.png")


if __name__ == "__main__":
    main()
