"""
Example 21 — Advanced Mathematical Intelligence: Neural PDE Solvers
Demonstrates solving heat and Poisson equations with the generalized PINN.
"""
import numpy as np
import matplotlib.pyplot as plt

from mnn.intelligence.neural_pde import NeuralPDESolver, heat_1d, poisson_2d


def main():
    print("=" * 60)
    print("  PILLAR 3: Neural PDE Solvers")
    print("=" * 60)

    # --- 3a. Heat Equation ---
    print("\n[3a] Solving Heat Equation: u_t = u_xx")
    problem = heat_1d(alpha=1.0, L=1.0, T=0.5)
    solver = NeuralPDESolver(problem, width=64, depth=4, lr=1e-3)
    solver.train(n_epochs=3000, n_collocation=1500, print_every=500)

    # Evaluate
    x_grid = np.linspace(0, 1, 50)
    t_grid = np.linspace(0, 0.5, 50)
    X, T_ = np.meshgrid(x_grid, t_grid)
    pts = np.column_stack([X.ravel(), T_.ravel()])

    u_pred = solver.predict(pts).reshape(50, 50)
    u_exact = problem.exact_solution(pts).reshape(50, 50)
    error = np.mean((u_pred - u_exact) ** 2)
    print(f"  MSE vs exact: {error:.8f}")

    # --- 3b. Poisson Equation ---
    print("\n[3b] Solving Poisson Equation: ∇²u = -2π²sin(πx)sin(πy)")

    def source_fn(xy):
        import torch
        return -2 * (np.pi ** 2) * torch.sin(np.pi * xy[:, 0:1]) * torch.sin(np.pi * xy[:, 1:2])

    problem2 = poisson_2d(source_fn=source_fn)
    problem2.exact_solution = lambda xy: (
        np.sin(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1])
    ).reshape(-1, 1)

    solver2 = NeuralPDESolver(problem2, width=64, depth=4, lr=1e-3)
    solver2.train(n_epochs=3000, n_collocation=2000, print_every=500)

    x2 = np.linspace(0, 1, 40)
    y2 = np.linspace(0, 1, 40)
    X2, Y2 = np.meshgrid(x2, y2)
    pts2 = np.column_stack([X2.ravel(), Y2.ravel()])
    u_pred2 = solver2.predict(pts2).reshape(40, 40)
    u_exact2 = problem2.exact_solution(pts2).reshape(40, 40)
    print(f"  MSE vs exact: {np.mean((u_pred2 - u_exact2) ** 2):.8f}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Heat equation
    im1 = axes[0, 0].pcolormesh(X, T_, u_exact, shading="auto", cmap="inferno")
    axes[0, 0].set_title("Heat: Exact"); axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("t")
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].pcolormesh(X, T_, u_pred, shading="auto", cmap="inferno")
    axes[0, 1].set_title("Heat: PINN"); axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("t")
    plt.colorbar(im2, ax=axes[0, 1])

    axes[0, 2].semilogy(solver.history["pde"], label="PDE")
    axes[0, 2].semilogy(solver.history.get("bc", []), label="BC")
    axes[0, 2].set_title("Heat: Training Loss"); axes[0, 2].legend()

    # Poisson equation
    im3 = axes[1, 0].pcolormesh(X2, Y2, u_exact2, shading="auto", cmap="viridis")
    axes[1, 0].set_title("Poisson: Exact"); plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].pcolormesh(X2, Y2, u_pred2, shading="auto", cmap="viridis")
    axes[1, 1].set_title("Poisson: PINN"); plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].pcolormesh(X2, Y2, np.abs(u_exact2 - u_pred2), shading="auto", cmap="Reds")
    axes[1, 2].set_title("Poisson: |Error|"); plt.colorbar(im5, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig("neural_pde_demo.png", dpi=150)
    plt.show()
    print("\nDone! Saved: neural_pde_demo.png")


if __name__ == "__main__":
    main()
