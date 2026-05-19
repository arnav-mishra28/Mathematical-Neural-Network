"""
Example 20 — Advanced Mathematical Intelligence: Group Theory Engine
Demonstrates neural group operations with algebraic constraints.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from mnn.intelligence.group_algebra import (
    NeuralGroupOperator, NeuralGroupTrainer,
    EquivariantNetwork, InvariantLearner,
)


def main():
    print("=" * 60)
    print("  PILLAR 2: Abstract Algebra Engine (Group Theory)")
    print("=" * 60)

    # --- 2a. Neural Abelian Group Operator (Z_n addition) ---
    print("\n[2a] Learning abelian group operation (Z_8 addition mod 8)...")

    n = 8
    elements_raw = np.eye(n, dtype=np.float32)  # one-hot encoding

    # Build supervised data: all (a,b) -> a+b mod n
    pairs, results = [], []
    for i in range(n):
        for j in range(n):
            pairs.append(np.concatenate([elements_raw[i], elements_raw[j]]))
            results.append(elements_raw[(i + j) % n])

    pairs = np.array(pairs, dtype=np.float32)
    results = np.array(results, dtype=np.float32)

    op = NeuralGroupOperator(element_dim=n, width=64, depth=3, abelian=True)
    trainer = NeuralGroupTrainer(op, lr=1e-3)
    history = trainer.train(
        elements_raw, pairs=pairs, results=results,
        n_epochs=1500, w_data=2.0, w_assoc=1.0, w_comm=1.0,
        print_every=300
    )

    # Verify commutativity
    a = torch.tensor(elements_raw[2:3])
    b = torch.tensor(elements_raw[5:6])
    with torch.no_grad():
        ab = op.operate(a, b)
        ba = op.operate(b, a)
    print(f"  f(a,b) ≈ f(b,a)? error = {torch.norm(ab - ba).item():.6f}")

    # --- 2b. Invariant Learning (rotation invariance) ---
    print("\n[2b] Learning rotation-invariant quantity...")

    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta),  np.cos(theta)]], dtype=np.float32)

    # Generate rotations as group actions
    n_rot = 8
    actions = [np.array([[np.cos(2*np.pi*k/n_rot), -np.sin(2*np.pi*k/n_rot)],
                          [np.sin(2*np.pi*k/n_rot),  np.cos(2*np.pi*k/n_rot)]],
                         dtype=np.float32) for k in range(n_rot)]

    # Data: 2D points, target = ||x||² (a rotation invariant)
    data = np.random.randn(500, 2).astype(np.float32)
    targets = (data[:, 0:1] ** 2 + data[:, 1:2] ** 2).astype(np.float32)

    inv_learner = InvariantLearner(
        input_dim=2, invariant_dim=1, group_actions=actions,
        width=64, depth=3, lr=1e-3
    )
    inv_learner.train(data, targets=targets, n_epochs=1000, print_every=200)

    # Test: I(Rx) ≈ I(x)?
    x_test = np.random.randn(100, 2).astype(np.float32)
    I_x = inv_learner.predict(x_test)
    I_Rx = inv_learner.predict(x_test @ R.T)
    inv_error = np.mean((I_x - I_Rx) ** 2)
    print(f"  Invariance error: {inv_error:.8f}")

    # --- 2c. Equivariant Network ---
    print("\n[2c] Testing equivariant network...")
    action_tensors = [torch.tensor(a) for a in actions]
    eq_net = EquivariantNetwork(2, 2, action_tensors, width=32, depth=2)
    x_test_t = torch.randn(50, 2)
    eq_err = eq_net.equivariance_error(x_test_t)
    print(f"  Equivariance error: {eq_err:.8f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].semilogy(history.get("total", []))
    axes[0].set_title("Group Operator Training Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Total Loss")

    axes[1].semilogy(history.get("commutativity", []), label="Commutativity")
    axes[1].semilogy(history.get("associativity", []), label="Associativity")
    axes[1].semilogy(history.get("identity", []), label="Identity")
    axes[1].set_title("Algebraic Constraint Losses")
    axes[1].set_xlabel("Epoch"); axes[1].legend()

    axes[2].scatter(targets.flatten(), inv_learner.predict(data).flatten(),
                    alpha=0.3, s=10)
    axes[2].plot([0, targets.max()], [0, targets.max()], "r--", lw=1)
    axes[2].set_xlabel("True ||x||²"); axes[2].set_ylabel("Predicted Invariant")
    axes[2].set_title("Invariant Learning")

    plt.tight_layout()
    plt.savefig("group_theory_demo.png", dpi=150)
    plt.show()
    print("\nDone! Saved: group_theory_demo.png")


if __name__ == "__main__":
    main()
