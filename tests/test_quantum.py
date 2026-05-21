"""Tests for the Quantum-Inspired Neural Math layer (mnn.quantum)."""
import numpy as np
import pytest


# ===== Hilbert Space Tests =====

class TestQuantumState:
    def test_normalization(self):
        from mnn.quantum.hilbert import QuantumState
        psi = QuantumState(np.array([3, 4j]))
        assert abs(np.linalg.norm(psi.amplitudes) - 1.0) < 1e-10

    def test_inner_product(self):
        from mnn.quantum.hilbert import QuantumState
        psi = QuantumState.computational_basis(0, 2)
        phi = QuantumState.computational_basis(1, 2)
        assert abs(psi.inner_product(phi)) < 1e-10  # orthogonal
        assert abs(psi.inner_product(psi) - 1.0) < 1e-10  # self

    def test_superposition(self):
        from mnn.quantum.hilbert import QuantumState
        psi = QuantumState.uniform_superposition(4)
        probs = psi.probabilities()
        assert np.allclose(probs, 0.25, atol=1e-10)

    def test_bloch_vector(self):
        from mnn.quantum.hilbert import QuantumState
        psi = QuantumState.computational_basis(0, 2)
        bv = psi.bloch_vector()
        assert np.allclose(bv, [0, 0, 1], atol=1e-10)

    def test_density_matrix(self):
        from mnn.quantum.hilbert import QuantumState
        psi = QuantumState.computational_basis(0, 2)
        rho = psi.density_matrix()
        assert rho.shape == (2, 2)
        assert abs(np.trace(rho) - 1.0) < 1e-10

    def test_tensor_product(self):
        from mnn.quantum.hilbert import QuantumState
        a = QuantumState.computational_basis(0, 2)
        b = QuantumState.computational_basis(1, 2)
        ab = a.tensor_product(b)
        assert ab.dim == 4
        assert np.allclose(ab.probabilities(), [0, 1, 0, 0], atol=1e-10)

    def test_evolve_unitary(self):
        from mnn.quantum.hilbert import QuantumState, QuantumGates
        psi = QuantumState.computational_basis(0, 2)
        psi_h = psi.evolve(QuantumGates.H)
        assert np.allclose(psi_h.probabilities(), [0.5, 0.5], atol=1e-10)

    def test_von_neumann_entropy(self):
        from mnn.quantum.hilbert import QuantumState, QuantumGates
        # Product state -> entropy = 0
        psi = QuantumState(np.array([1, 0, 0, 0], dtype=complex))
        assert abs(psi.von_neumann_entropy((2, 2))) < 1e-10
        # Maximally entangled -> entropy = 1
        bell = QuantumState(np.array([1, 0, 0, 1], dtype=complex))
        assert abs(bell.von_neumann_entropy((2, 2)) - 1.0) < 1e-5


class TestHilbertSpace:
    def test_random_unitary(self):
        from mnn.quantum.hilbert import HilbertSpace
        H = HilbertSpace(4)
        U = H.random_unitary(seed=42)
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-10)

    def test_random_hermitian(self):
        from mnn.quantum.hilbert import HilbertSpace
        H = HilbertSpace(4)
        A = H.random_hermitian(seed=42)
        assert np.allclose(A, A.conj().T, atol=1e-10)

    def test_commutator(self):
        from mnn.quantum.hilbert import HilbertSpace
        H = HilbertSpace(2)
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        comm = H.commutator(sx, sy)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        assert np.allclose(comm, 2j * sz, atol=1e-10)


class TestQuantumGates:
    def test_hadamard_unitary(self):
        from mnn.quantum.hilbert import QuantumGates
        H = QuantumGates.H
        assert np.allclose(H @ H.conj().T, np.eye(2), atol=1e-10)

    def test_pauli_anticommutation(self):
        from mnn.quantum.hilbert import QuantumGates
        XY = QuantumGates.X @ QuantumGates.Y + QuantumGates.Y @ QuantumGates.X
        assert np.allclose(XY, np.zeros((2, 2)), atol=1e-10)


# ===== Complex NN Tests =====

class TestComplexNN:
    def test_complex_linear_shape(self):
        import torch
        from mnn.quantum.complex_nn import ComplexLinear
        cl = ComplexLinear(4, 8)
        r, i = cl(torch.randn(5, 4), torch.randn(5, 4))
        assert r.shape == (5, 8)
        assert i.shape == (5, 8)

    def test_complex_network_forward(self):
        import torch
        from mnn.quantum.complex_nn import ComplexNeuralNetwork
        net = ComplexNeuralNetwork(3, 2, width=16, depth=2)
        out = net(torch.randn(10, 3))
        assert out.shape == (10, 2)

    def test_magnitude_phase(self):
        import torch
        from mnn.quantum.complex_nn import ComplexNeuralNetwork
        net = ComplexNeuralNetwork(2, 1, width=16, depth=1)
        mag, phase = net.magnitude_phase(torch.randn(5, 2))
        assert mag.shape[0] == 5
        assert phase.shape[0] == 5


# ===== Unitary Tests =====

class TestUnitary:
    def test_unitary_layer_preserves_unitarity(self):
        from mnn.quantum.unitary import UnitaryLayer
        ul = UnitaryLayer(4)
        assert ul.unitarity_error() < 1e-5

    def test_unitary_network_forward(self):
        import torch
        from mnn.quantum.unitary import UnitaryNetwork
        un = UnitaryNetwork(3, 2, hidden_dim=16, n_blocks=2)
        out = un(torch.randn(10, 3))
        assert out.shape == (10, 2)

    def test_parameterized_unitary(self):
        import torch
        from mnn.quantum.unitary import ParameterizedUnitary
        pu = ParameterizedUnitary(4)
        U = pu.unitary_matrix()
        err = torch.norm(U.conj().T @ U - torch.eye(4, dtype=torch.cfloat))
        assert err.item() < 1e-4


# ===== Attention Tests =====

class TestQuantumAttention:
    def test_multi_head_shape(self):
        import torch
        from mnn.quantum.attention import QuantumMultiHeadAttention
        qmha = QuantumMultiHeadAttention(16, n_heads=4)
        x = torch.randn(2, 5, 16)
        out, weights = qmha(x)
        assert out.shape == (2, 5, 16)
        assert len(weights) == 4

    def test_transformer_forward(self):
        import torch
        from mnn.quantum.attention import QuantumTransformer
        qt = QuantumTransformer(8, 2, d_model=16, n_heads=4, n_layers=2)
        out, _ = qt(torch.randn(10, 8))
        assert out.shape == (10, 2)


# ===== Geometric Tests =====

class TestQuantumGeometric:
    def test_fubini_study_distance(self):
        from mnn.quantum.geometric import FubiniStudyMetric
        psi = np.array([1, 0], dtype=complex)
        phi = np.array([0, 1], dtype=complex)
        d = FubiniStudyMetric.distance(psi, phi)
        assert abs(d - np.pi / 2) < 1e-10

    def test_fubini_study_zero_distance(self):
        from mnn.quantum.geometric import FubiniStudyMetric
        psi = np.array([1, 1j], dtype=complex)
        assert FubiniStudyMetric.distance(psi, psi) < 1e-6

    def test_berry_phase_trivial(self):
        from mnn.quantum.geometric import FubiniStudyMetric
        psi = np.array([1, 0], dtype=complex)
        bp = FubiniStudyMetric.berry_phase([psi, psi, psi])
        assert abs(bp) < 1e-10

    def test_quantum_embedding_shape(self):
        import torch
        from mnn.quantum.geometric import QuantumEmbedding
        emb = QuantumEmbedding(4, 8)
        r, i = emb(torch.randn(10, 4))
        assert r.shape == (10, 8)
        assert i.shape == (10, 8)

    def test_geometric_network_forward(self):
        import torch
        from mnn.quantum.geometric import QuantumGeometricNetwork
        qgn = QuantumGeometricNetwork(3, 1, state_dim=8, n_geo_layers=2)
        out = qgn(torch.randn(5, 3))
        assert out.shape == (5, 1)


# ===== Quantum Chaos Tests =====

class TestQuantumChaos:
    def test_goe_hermitian(self):
        from mnn.quantum.chaos import RandomMatrixEnsemble
        H = RandomMatrixEnsemble.goe(50, seed=42)
        assert np.allclose(H, H.T, atol=1e-10)

    def test_gue_hermitian(self):
        from mnn.quantum.chaos import RandomMatrixEnsemble
        H = RandomMatrixEnsemble.gue(50, seed=42)
        assert np.allclose(H, H.conj().T, atol=1e-10)

    def test_cue_unitary(self):
        from mnn.quantum.chaos import RandomMatrixEnsemble
        U = RandomMatrixEnsemble.circular_unitary(20, seed=42)
        assert np.allclose(U @ U.conj().T, np.eye(20), atol=1e-8)

    def test_spectral_classifier(self):
        from mnn.quantum.chaos import RandomMatrixEnsemble, SpectralAnalyzer
        goe = RandomMatrixEnsemble.goe(200, seed=42)
        evals = SpectralAnalyzer.eigenvalues(goe)
        stats = SpectralAnalyzer.classify_dynamics(evals)
        assert "chaotic" in stats["classification"]

    def test_level_spacing_ratio(self):
        from mnn.quantum.chaos import RandomMatrixEnsemble, SpectralAnalyzer
        goe = RandomMatrixEnsemble.goe(200, seed=42)
        evals = SpectralAnalyzer.eigenvalues(goe)
        ratios = SpectralAnalyzer.level_spacing_ratio(evals)
        mean_r = np.mean(ratios)
        assert 0.4 < mean_r < 0.7  # should be ~0.53 for GOE

    def test_kicked_top(self):
        from mnn.quantum.chaos import QuantumKickedTop
        qkt = QuantumKickedTop(j=5, k=3.0)
        psi0 = np.zeros(qkt.dim, dtype=complex)
        psi0[0] = 1.0
        traj = qkt.evolve(psi0, n_kicks=10)
        assert traj.shape == (11, qkt.dim)
        # Norm preserved
        assert abs(np.linalg.norm(traj[-1]) - 1.0) < 1e-10

    def test_page_entropy(self):
        from mnn.quantum.chaos import QuantumEntanglementDynamics
        S_page = QuantumEntanglementDynamics.page_entropy(2, 2)
        assert S_page > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
