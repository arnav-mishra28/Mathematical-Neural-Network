"""mnn.quantum.chaos — Quantum chaos module.

Part 6: Merge chaos + quantum math. Studies nonlinear quantum-like evolution,
chaotic state transitions, spectral dynamics, random matrix theory,
and level spacing statistics (GOE/GUE/Poisson).
"""
from __future__ import annotations
import numpy as np
from scipy import linalg as sla
from typing import Optional, Dict, List, Tuple


class RandomMatrixEnsemble:
    """Random matrix theory: GOE, GUE, Wishart ensembles.

    Central to quantum chaos — spectral statistics distinguish
    integrable (Poisson) from chaotic (Wigner-Dyson) dynamics.
    """
    @staticmethod
    def goe(n: int, seed: Optional[int] = None) -> np.ndarray:
        """Gaussian Orthogonal Ensemble (time-reversal symmetric)."""
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n, n))
        return (A + A.T) / (2 * np.sqrt(n))

    @staticmethod
    def gue(n: int, seed: Optional[int] = None) -> np.ndarray:
        """Gaussian Unitary Ensemble (no time-reversal symmetry)."""
        rng = np.random.default_rng(seed)
        A = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
        return (A + A.conj().T) / (2 * np.sqrt(n))

    @staticmethod
    def wishart(n: int, m: int, seed: Optional[int] = None) -> np.ndarray:
        """Wishart ensemble: W = X^T X / m."""
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((m, n))
        return X.T @ X / m

    @staticmethod
    def circular_unitary(n: int, seed: Optional[int] = None) -> np.ndarray:
        """Circular Unitary Ensemble (random unitary matrix)."""
        rng = np.random.default_rng(seed)
        Z = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
        Q, R = np.linalg.qr(Z)
        D = np.diag(R)
        return Q * (D / np.abs(D))


class SpectralAnalyzer:
    """Analyze spectral statistics of quantum Hamiltonians.

    Level spacing distributions distinguish:
      - Integrable systems → Poisson statistics: P(s) = exp(-s)
      - Chaotic systems   → Wigner-Dyson: P(s) = (pi*s/2) exp(-pi*s^2/4)
    """
    @staticmethod
    def eigenvalues(H: np.ndarray) -> np.ndarray:
        if np.allclose(H, H.conj().T):
            return np.linalg.eigvalsh(H)
        return np.sort(np.real(np.linalg.eigvals(H)))

    @staticmethod
    def level_spacings(eigenvalues: np.ndarray) -> np.ndarray:
        """Normalized nearest-neighbor level spacings."""
        sorted_e = np.sort(np.real(eigenvalues))
        spacings = np.diff(sorted_e)
        mean_s = np.mean(spacings)
        if mean_s > 0:
            spacings /= mean_s
        return spacings

    @staticmethod
    def level_spacing_ratio(eigenvalues: np.ndarray) -> np.ndarray:
        """r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}).

        <r> ~ 0.386 for Poisson, <r> ~ 0.530 for GOE, <r> ~ 0.603 for GUE.
        """
        spacings = np.diff(np.sort(np.real(eigenvalues)))
        ratios = np.minimum(spacings[:-1], spacings[1:]) / (np.maximum(spacings[:-1], spacings[1:]) + 1e-15)
        return ratios

    @staticmethod
    def spectral_rigidity(eigenvalues: np.ndarray, L_max: float = 10.0,
                           n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Delta_3(L) spectral rigidity statistic."""
        E = np.sort(np.real(eigenvalues))
        N = len(E)
        # Unfold spectrum
        mean_spacing = (E[-1] - E[0]) / (N - 1)
        E_unf = (E - E[0]) / mean_spacing

        L_values = np.linspace(0.5, L_max, n_points)
        delta3 = np.zeros(n_points)

        for i, L in enumerate(L_values):
            best = []
            for start_idx in range(0, N - 2, max(1, N // 20)):
                e0 = E_unf[start_idx]
                window = E_unf[(E_unf >= e0) & (E_unf < e0 + L)]
                if len(window) < 2:
                    continue
                n_w = np.arange(len(window)) + 1
                # Least squares fit to staircase
                A = np.column_stack([np.ones(len(window)), window - e0])
                if len(window) >= 2:
                    coeff = np.linalg.lstsq(A, n_w, rcond=None)[0]
                    fitted = A @ coeff
                    val = np.mean((n_w - fitted) ** 2) / L
                    best.append(val)
            delta3[i] = np.mean(best) if best else 0

        return L_values, delta3

    @staticmethod
    def classify_dynamics(eigenvalues: np.ndarray) -> Dict:
        """Classify system as integrable or chaotic from spectral statistics."""
        ratios = SpectralAnalyzer.level_spacing_ratio(eigenvalues)
        mean_r = float(np.mean(ratios))
        spacings = SpectralAnalyzer.level_spacings(eigenvalues)

        if mean_r < 0.45:
            classification = "integrable (Poisson)"
        elif mean_r < 0.56:
            classification = "chaotic (GOE)"
        else:
            classification = "chaotic (GUE)"

        return {
            "classification": classification,
            "mean_r": mean_r,
            "mean_spacing": float(np.mean(spacings)),
            "std_spacing": float(np.std(spacings)),
            "n_levels": len(eigenvalues),
        }


class QuantumKickedTop:
    """Quantum kicked top — paradigmatic model of quantum chaos.

    H = p * J_z^2 / (2j) + k * J_y * sum_n delta(t - n)
    """
    def __init__(self, j: float = 10, k: float = 3.0, p: float = np.pi / 2):
        self.j = j
        self.dim = int(2 * j + 1)
        self.k = k
        self.p = p
        self._build_operators()

    def _build_operators(self):
        j = self.j
        m_vals = np.arange(-j, j + 1)
        self.Jz = np.diag(m_vals).astype(complex)
        Jp = np.zeros((self.dim, self.dim), dtype=complex)
        Jm = np.zeros((self.dim, self.dim), dtype=complex)
        for idx in range(self.dim - 1):
            m = m_vals[idx]
            Jp[idx + 1, idx] = np.sqrt(j * (j + 1) - m * (m + 1))
            Jm[idx, idx + 1] = np.sqrt(j * (j + 1) - (m + 1) * m)
        self.Jx = (Jp + Jm) / 2
        self.Jy = (Jp - Jm) / (2j)

    def floquet_operator(self) -> np.ndarray:
        """Single-period evolution operator."""
        U_kick = sla.expm(-1j * self.k * self.Jy)
        U_free = sla.expm(-1j * self.p * self.Jz @ self.Jz / (2 * self.j))
        return U_kick @ U_free

    def evolve(self, state: np.ndarray, n_kicks: int) -> np.ndarray:
        """Evolve state for n_kicks periods. Returns trajectory of states."""
        U = self.floquet_operator()
        trajectory = [state.copy()]
        psi = state.copy()
        for _ in range(n_kicks):
            psi = U @ psi
            psi /= np.linalg.norm(psi)
            trajectory.append(psi.copy())
        return np.array(trajectory)

    def husimi(self, state: np.ndarray, n_grid: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Husimi Q function on the Bloch sphere."""
        theta = np.linspace(0, np.pi, n_grid)
        phi = np.linspace(0, 2 * np.pi, n_grid)
        Q = np.zeros((n_grid, n_grid))

        for i, t in enumerate(theta):
            for k, p in enumerate(phi):
                coherent = self._coherent_state(t, p)
                Q[i, k] = np.abs(np.vdot(coherent, state)) ** 2

        return theta, phi, Q

    def _coherent_state(self, theta: float, phi: float) -> np.ndarray:
        """Spin coherent state |theta, phi>."""
        j = self.j
        state = np.zeros(self.dim, dtype=complex)
        for idx, m in enumerate(np.arange(-j, j + 1)):
            from scipy.special import comb
            coeff = np.sqrt(comb(int(2*j), int(j+m), exact=True))
            state[idx] = coeff * (np.cos(theta/2)**(j+m)) * (np.sin(theta/2)**(j-m)) * np.exp(1j * m * phi)
        state /= np.linalg.norm(state)
        return state

    def spectral_statistics(self) -> Dict:
        """Analyze the Floquet operator spectrum."""
        U = self.floquet_operator()
        eigenphases = np.sort(np.angle(np.linalg.eigvals(U)))
        return SpectralAnalyzer.classify_dynamics(eigenphases)


class QuantumLyapunov:
    """Quantum Lyapunov exponent via out-of-time-order correlators (OTOC).

    F(t) = <W(t)^dag V^dag W(t) V> ~ 1 - epsilon * exp(2 lambda_Q t)
    """
    @staticmethod
    def otoc(H: np.ndarray, V: np.ndarray, W: np.ndarray,
             times: np.ndarray, beta: float = 0.0) -> np.ndarray:
        """Compute OTOC F(t) = Tr[W(t)^dag V^dag W(t) V rho] / Tr[rho]."""
        n = H.shape[0]
        if beta > 0:
            rho = sla.expm(-beta * H)
            rho /= np.trace(rho)
        else:
            rho = np.eye(n) / n

        results = []
        for t in times:
            U = sla.expm(-1j * H * t)
            Ud = U.conj().T
            Wt = Ud @ W @ U
            comm = Wt.conj().T @ V.conj().T @ Wt @ V
            F = np.real(np.trace(comm @ rho))
            results.append(F)
        return np.array(results)

    @staticmethod
    def estimate_lyapunov(otoc_values: np.ndarray, times: np.ndarray) -> float:
        """Estimate quantum Lyapunov exponent from OTOC decay."""
        # F(t) ~ 1 - eps * exp(2*lambda*t)
        # log(1 - F(t)) ~ log(eps) + 2*lambda*t
        deviation = 1 - otoc_values
        mask = deviation > 1e-10
        if np.sum(mask) < 3:
            return 0.0
        log_dev = np.log(deviation[mask])
        t_masked = times[mask]
        # Linear fit
        coeffs = np.polyfit(t_masked, log_dev, 1)
        return float(coeffs[0] / 2)


class QuantumEntanglementDynamics:
    """Track entanglement growth under chaotic evolution.

    In chaotic systems, entanglement grows linearly then saturates.
    In integrable systems, it grows logarithmically.
    """
    @staticmethod
    def entanglement_entropy_evolution(H: np.ndarray, psi0: np.ndarray,
                                        subsystem_dims: Tuple[int, int],
                                        times: np.ndarray) -> np.ndarray:
        d1, d2 = subsystem_dims
        entropies = []
        for t in times:
            U = sla.expm(-1j * H * t)
            psi = U @ psi0
            psi /= np.linalg.norm(psi)
            rho = np.outer(psi, psi.conj()).reshape(d1, d2, d1, d2)
            rho_A = np.trace(rho, axis1=1, axis2=3)
            evals = np.real(np.linalg.eigvalsh(rho_A))
            evals = evals[evals > 1e-15]
            S = -np.sum(evals * np.log2(evals))
            entropies.append(S)
        return np.array(entropies)

    @staticmethod
    def page_entropy(d1: int, d2: int) -> float:
        """Page's formula for average entanglement entropy of random states."""
        d = d1 * d2
        if d1 <= d2:
            return sum(1.0/k for k in range(d2+1, d+1)) / np.log(2) - (d1-1) / (2*d2*np.log(2))
        return QuantumEntanglementDynamics.page_entropy(d2, d1)

    @staticmethod
    def classify_entanglement_growth(entropies: np.ndarray,
                                      times: np.ndarray) -> Dict:
        """Classify entanglement growth rate."""
        mid = len(times) // 2
        early_slope = np.polyfit(times[:mid], entropies[:mid], 1)[0]
        late_slope = np.polyfit(times[mid:], entropies[mid:], 1)[0]

        if early_slope > 0.1 and late_slope < early_slope * 0.3:
            growth = "linear then saturating (chaotic)"
        elif early_slope < 0.05:
            growth = "logarithmic (integrable)"
        else:
            growth = "intermediate"

        return {
            "growth_type": growth,
            "early_slope": float(early_slope),
            "late_slope": float(late_slope),
            "max_entropy": float(np.max(entropies)),
        }
