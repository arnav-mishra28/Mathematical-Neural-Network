"""
Example 22 — Scientific Discovery Engine
Discovers governing equations from trajectory data.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mnn.intelligence.discovery import HybridDiscovery, ScientificDiscoveryEngine
from mnn.chaos.attractors import LorenzAttractor


def main():
    print("=" * 60)
    print("  PILLAR 4: Scientific Discovery Engine")
    print("=" * 60)

    # --- 4a. Discover Lorenz equations ---
    print("\n[4a] Discovering Lorenz equations from data...")
    lorenz = LorenzAttractor(sigma=10, rho=28, beta=8/3)
    trajectory = lorenz.integrate(t_span=(0, 20), dt=0.01)
    t = np.arange(len(trajectory)) * 0.01

    discovery = HybridDiscovery(state_dim=3, poly_order=2, threshold=0.5)
    result = discovery.discover(
        t, trajectory, n_smooth_epochs=2000,
        var_names=["dx/dt", "dy/dt", "dz/dt"], verbose=True
    )

    # --- 4b. Damped oscillator ---
    print("\n[4b] Discovering damped oscillator...")
    sol = solve_ivp(lambda t,s: [s[1], -s[0]-0.5*s[1]], (0,20), [1,0],
                    t_eval=np.linspace(0,20,2000))
    d2 = HybridDiscovery(state_dim=2, poly_order=2, threshold=0.05)
    d2.discover(sol.t, sol.y.T, n_smooth_epochs=2000,
                var_names=["dx/dt","dy/dt"], verbose=True)

    # --- 4c. Auto-discovery ---
    print("\n[4c] Automated multi-threshold sweep...")
    engine = ScientificDiscoveryEngine(state_dim=3)
    engine.auto_discover(t[:1500], trajectory[:1500],
                         thresholds=[0.1,0.3,0.5,1.0], poly_orders=[2],
                         var_names=["dx/dt","dy/dt","dz/dt"], verbose=True)

    # Plot
    fig = plt.figure(figsize=(14,5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot(trajectory[:1500,0], trajectory[:1500,1], trajectory[:1500,2], lw=0.4)
    ax1.set_title("Lorenz (Input)")
    ax2 = fig.add_subplot(132)
    ax2.plot(sol.y[0], sol.y[1], lw=0.5)
    ax2.set_title("Oscillator Phase"); ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax3 = fig.add_subplot(133)
    if result["coefficients"] is not None:
        ax3.imshow(np.abs(result["coefficients"]), aspect="auto", cmap="hot")
        ax3.set_title("|Ξ| Sparsity"); ax3.set_xlabel("State"); ax3.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig("scientific_discovery_demo.png", dpi=150)
    plt.show()
    print("\nSaved: scientific_discovery_demo.png")

if __name__ == "__main__":
    main()
