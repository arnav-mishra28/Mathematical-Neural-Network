"""mnn.quantum.hilbert — Hilbert space layer and quantum state representations.

Part 1: The foundation — complex-valued state spaces |psi> in C^n.
Instead of flat real vectors, MNN now represents states as complex
amplitudes with magnitude and phase, enabling interference and superposition.
"""
from __future__ import annotations
import torch, torch.nn as nn, numpy as np
from typing import Optional, List, Tuple, Dict


class QuantumState:
    """A quantum state |psi> = sum_i c_i |i> in a finite-dimensional Hilbert space.

    Stores complex amplitudes, supports inner products, tensor products,
    measurement probabilities, density matrices, and entanglement entropy.
    """
    def __init__(self, amplitudes: np.ndarray, basis_labels: Optional[List[str]] = None):
        self.amplitudes = np.asarray(amplitudes, dtype=complex)
        self.dim = len(self.amplitudes)
        self.basis_labels = basis_labels or [f"|{i}>" for i in range(self.dim)]
        self._normalize()

    def _normalize(self):
        norm = np.linalg.norm(self.amplitudes)
        if norm > 1e-15:
            self.amplitudes /= norm

    @classmethod
    def computational_basis(cls, i: int, dim: int) -> "QuantumState":
        amps = np.zeros(dim, dtype=complex)
        amps[i] = 1.0
        return cls(amps)

    @classmethod
    def uniform_superposition(cls, dim: int) -> "QuantumState":
        return cls(np.ones(dim, dtype=complex) / np.sqrt(dim))

    @classmethod
    def random_state(cls, dim: int, seed: Optional[int] = None) -> "QuantumState":
        rng = np.random.default_rng(seed)
        real = rng.standard_normal(dim)
        imag = rng.standard_normal(dim)
        return cls(real + 1j * imag)

    @classmethod
    def from_bloch(cls, theta: float, phi: float) -> "QuantumState":
        """Create a qubit state from Bloch sphere angles."""
        return cls(np.array([np.cos(theta / 2),
                             np.exp(1j * phi) * np.sin(theta / 2)]))

    def inner_product(self, other: "QuantumState") -> complex:
        """<self|other>"""
        return complex(np.vdot(self.amplitudes, other.amplitudes))

    def overlap(self, other: "QuantumState") -> float:
        """|<self|other>|^2"""
        return abs(self.inner_product(other)) ** 2

    def probabilities(self) -> np.ndarray:
        return np.abs(self.amplitudes) ** 2

    def measure(self, n_shots: int = 1) -> np.ndarray:
        probs = self.probabilities()
        return np.random.choice(self.dim, size=n_shots, p=probs)

    def density_matrix(self) -> np.ndarray:
        psi = self.amplitudes.reshape(-1, 1)
        return psi @ psi.conj().T

    def von_neumann_entropy(self, subsystem_dims: Tuple[int, int]) -> float:
        """Entanglement entropy via partial trace."""
        d1, d2 = subsystem_dims
        if d1 * d2 != self.dim:
            raise ValueError(f"Subsystem dims {d1}x{d2} != {self.dim}")
        rho = self.density_matrix().reshape(d1, d2, d1, d2)
        rho_A = np.trace(rho, axis1=1, axis2=3)
        evals = np.real(np.linalg.eigvalsh(rho_A))
        evals = evals[evals > 1e-15]
        return float(-np.sum(evals * np.log2(evals)))

    def expectation(self, operator: np.ndarray) -> complex:
        """<psi|O|psi>"""
        return complex(self.amplitudes.conj() @ operator @ self.amplitudes)

    def evolve(self, unitary: np.ndarray) -> "QuantumState":
        """Apply unitary U: |psi> -> U|psi>"""
        return QuantumState(unitary @ self.amplitudes, self.basis_labels)

    def tensor_product(self, other: "QuantumState") -> "QuantumState":
        """|self> tensor |other>"""
        combined = np.kron(self.amplitudes, other.amplitudes)
        labels = [f"{a}{b}" for a in self.basis_labels for b in other.basis_labels]
        return QuantumState(combined, labels)

    def fidelity(self, other: "QuantumState") -> float:
        return self.overlap(other)

    def bloch_vector(self) -> np.ndarray:
        """Bloch vector for qubit states."""
        if self.dim != 2:
            raise ValueError("Bloch vector only for qubits (dim=2)")
        rho = self.density_matrix()
        sx = np.array([[0, 1], [1, 0]])
        sy = np.array([[0, -1j], [1j, 0]])
        sz = np.array([[1, 0], [0, -1]])
        return np.real(np.array([np.trace(rho @ sx),
                                  np.trace(rho @ sy),
                                  np.trace(rho @ sz)]))

    def __repr__(self):
        top = np.argsort(-np.abs(self.amplitudes))[:3]
        terms = [f"{self.amplitudes[i]:.3f}{self.basis_labels[i]}" for i in top]
        return f"QuantumState(dim={self.dim}, {' + '.join(terms)}...)"


class HilbertSpace:
    """A finite-dimensional Hilbert space H = C^n.

    Provides operator algebra, spectral decomposition, and state manipulation.
    """
    def __init__(self, dim: int, name: str = "H"):
        self.dim = dim
        self.name = name

    def zero_state(self) -> QuantumState:
        return QuantumState.computational_basis(0, self.dim)

    def identity(self) -> np.ndarray:
        return np.eye(self.dim, dtype=complex)

    def random_unitary(self, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        Z = (rng.standard_normal((self.dim, self.dim))
             + 1j * rng.standard_normal((self.dim, self.dim))) / np.sqrt(2)
        Q, R = np.linalg.qr(Z)
        D = np.diag(R)
        Q *= (D / np.abs(D))
        return Q

    def random_hermitian(self, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        A = (rng.standard_normal((self.dim, self.dim))
             + 1j * rng.standard_normal((self.dim, self.dim)))
        return (A + A.conj().T) / 2

    def projector(self, state: QuantumState) -> np.ndarray:
        return state.density_matrix()

    def commutator(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A @ B - B @ A

    def anticommutator(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A @ B + B @ A

    def spectral_decomposition(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        evals, evecs = np.linalg.eigh(H)
        return evals, evecs

    def tensor_product_space(self, other: "HilbertSpace") -> "HilbertSpace":
        return HilbertSpace(self.dim * other.dim, f"{self.name}x{other.name}")

    def partial_trace(self, rho: np.ndarray, dims: Tuple[int, int],
                      trace_out: int = 1) -> np.ndarray:
        d1, d2 = dims
        rho_r = rho.reshape(d1, d2, d1, d2)
        if trace_out == 1:
            return np.trace(rho_r, axis1=1, axis2=3)
        return np.trace(rho_r, axis1=0, axis2=2)

    def __repr__(self):
        return f"HilbertSpace({self.name}, dim={self.dim})"


# Standard quantum gates as numpy arrays
class QuantumGates:
    """Standard quantum gate matrices."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)

    @staticmethod
    def Rx(theta: float) -> np.ndarray:
        return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                         [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)

    @staticmethod
    def Ry(theta: float) -> np.ndarray:
        return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                         [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)

    @staticmethod
    def Rz(phi: float) -> np.ndarray:
        return np.array([[np.exp(-1j*phi/2), 0],
                         [0, np.exp(1j*phi/2)]], dtype=complex)

    @staticmethod
    def phase_gate(phi: float) -> np.ndarray:
        return np.array([[1, 0], [0, np.exp(1j*phi)]], dtype=complex)
