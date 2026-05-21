"""
Example 26 — Quantum-Inspired Neural Math: Hilbert Space & Complex Networks
Parts 1-3: Quantum states, complex-valued NNs, unitary transformations.
"""
import numpy as np


def main():
    print("=" * 65)
    print("  QUANTUM-INSPIRED NEURAL MATH — Parts 1-3")
    print("=" * 65)

    # ---- Part 1: Hilbert Space ----
    print("\n[1] Hilbert Space Layer")
    from mnn.quantum.hilbert import QuantumState, HilbertSpace, QuantumGates

    # Create quantum states
    psi = QuantumState.uniform_superposition(4)
    phi = QuantumState.computational_basis(0, 4)
    print(f"  |psi> = {psi}")
    print(f"  |phi> = {phi}")
    print(f"  <phi|psi> = {psi.inner_product(phi):.4f}")
    print(f"  |<phi|psi>|^2 = {psi.overlap(phi):.4f}")
    print(f"  Probabilities: {psi.probabilities()}")

    # Bloch sphere
    qubit = QuantumState.from_bloch(np.pi/3, np.pi/4)
    print(f"  Bloch qubit: {qubit.bloch_vector()}")

    # Apply Hadamard gate
    H_gate = QuantumGates.H
    q0 = QuantumState.computational_basis(0, 2)
    q_hadamard = q0.evolve(H_gate)
    print(f"  H|0> = {q_hadamard.probabilities()} (should be [0.5, 0.5])")

    # Tensor product
    bell_input = q_hadamard.tensor_product(QuantumState.computational_basis(0, 2))
    bell_state = bell_input.evolve(QuantumGates.CNOT)
    print(f"  Bell state entropy: {bell_state.von_neumann_entropy((2, 2)):.4f}")

    # Hilbert space operations
    H = HilbertSpace(4)
    hamiltonian = H.random_hermitian(seed=42)
    evals, evecs = H.spectral_decomposition(hamiltonian)
    print(f"  Random Hamiltonian eigenvalues: {np.round(evals, 3)}")

    # ---- Part 2: Complex Neural Networks ----
    print("\n[2] Complex-Valued Neural Networks")
    import torch
    from mnn.quantum.complex_nn import ComplexNeuralNetwork, ComplexTrainer

    # Learn f(x) = sin(x) using complex network
    x = np.linspace(-3, 3, 500).reshape(-1, 1).astype(np.float32)
    y = np.sin(x).astype(np.float32)

    cnn = ComplexNeuralNetwork(1, 1, width=32, depth=3, activation="modrelu")
    print(f"  {cnn}")
    trainer = ComplexTrainer(cnn, lr=1e-3)
    trainer.train(x, y, n_epochs=1000, verbose=True, print_every=500)

    pred = cnn.predict_numpy(x)
    mse = np.mean((pred - y)**2)
    print(f"  sin(x) MSE: {mse:.8f}")

    # Analyze magnitude and phase
    mag, phase = cnn.magnitude_phase(torch.tensor(x))
    print(f"  Internal magnitude range: [{mag.min():.3f}, {mag.max():.3f}]")
    print(f"  Internal phase range: [{phase.min():.3f}, {phase.max():.3f}]")

    # ---- Part 3: Unitary Transformations ----
    print("\n[3] Unitary Transformations")
    from mnn.quantum.unitary import UnitaryLayer, UnitaryNetwork, UnitaryTrainer

    # Verify unitarity
    ul = UnitaryLayer(8)
    print(f"  UnitaryLayer(8) unitarity error: {ul.unitarity_error():.10f}")

    # Train unitary network
    x2 = np.random.randn(500, 3).astype(np.float32)
    y2 = np.column_stack([np.sin(x2[:, 0]), np.cos(x2[:, 1])]).astype(np.float32)

    un = UnitaryNetwork(3, 2, hidden_dim=32, n_blocks=3)
    ut = UnitaryTrainer(un, lr=1e-3, unitarity_weight=0.01)
    ut.train(x2, y2, n_epochs=800, verbose=True, print_every=400)

    print(f"  Total unitarity error: {un.total_unitarity_error():.8f}")
    print(f"  Parameters: {un.count_parameters():,}")

    print("\n  Parts 1-3 complete!")


if __name__ == "__main__":
    main()
