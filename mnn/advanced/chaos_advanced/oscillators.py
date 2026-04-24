"""
mnn.advanced.chaos_advanced.oscillators
=========================================
Coupled oscillator systems and synchronization analysis.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Optional

class CoupledOscillators:
    """Coupled nonlinear oscillator systems."""

    @staticmethod
    def kuramoto_model(N: int = 10, K: float = 2.0,
                        t_end: float = 50., dt: float = 0.05,
                        seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kuramoto model: dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
        Returns (t_array, theta_array of shape (n_steps, N)).
        Order parameter r = |mean(exp(iθ))| measures synchronization.
        """
        rng    = np.random.default_rng(seed)
        omegas = rng.normal(0, 1, N)   # natural frequencies
        theta0 = rng.uniform(0, 2*np.pi, N)

        def ode(t, theta):
            diffs = theta[:, None] - theta[None, :]   # θⱼ - θᵢ
            return omegas + (K/N)*np.sum(np.sin(diffs), axis=1)

        t_eval = np.arange(0, t_end, dt)
        sol    = solve_ivp(ode, [0, t_end], theta0, t_eval=t_eval, method='RK45', rtol=1e-6)
        return sol.t, sol.y.T

    @staticmethod
    def order_parameter(theta: np.ndarray) -> np.ndarray:
        """r(t) = |mean(exp(iθ))| — Kuramoto order parameter."""
        return np.abs(np.mean(np.exp(1j*theta), axis=1))

    @staticmethod
    def van_der_pol(mu: float = 1.0, t_end: float = 30., dt: float = 0.01) -> np.ndarray:
        """
        Van der Pol oscillator: ẍ − μ(1−x²)ẋ + x = 0.
        μ=0 → harmonic; μ>0 → limit cycle; μ>>1 → relaxation oscillations.
        """
        def ode(t, s):
            x, v = s
            return [v, mu*(1-x**2)*v - x]
        sol = solve_ivp(ode, [0,t_end], [2.,0.], t_eval=np.arange(0,t_end,dt), rtol=1e-8)
        return sol.y.T

    @staticmethod
    def duffing_oscillator(alpha: float = 1., beta: float = -1.,
                            gamma: float = 0.3, omega: float = 1.2,
                            delta: float = 0.15,
                            t_end: float = 100., dt: float = 0.01) -> np.ndarray:
        """
        Duffing oscillator (forced, damped):
        ẍ + δẋ + αx + βx³ = γcos(ωt)
        With correct parameters → strange attractor.
        """
        def ode(t, s):
            x, v = s
            return [v, gamma*np.cos(omega*t) - delta*v - alpha*x - beta*x**3]
        sol = solve_ivp(ode, [0,t_end], [0.,0.], t_eval=np.arange(0,t_end,dt), rtol=1e-8)
        return sol.y.T

    @staticmethod
    def coupled_lorenz(N: int = 3, coupling: float = 0.1,
                        t_end: float = 30., dt: float = 0.01) -> np.ndarray:
        """
        N coupled Lorenz systems.
        dxᵢ/dt = σ(yᵢ-xᵢ) + c(xᵢ₋₁ - 2xᵢ + xᵢ₊₁)  [ring topology]
        Returns trajectory of shape (n_steps, 3N).
        """
        sigma, rho, beta = 10., 28., 8/3
        def ode(t, state):
            state = state.reshape(N, 3); ds = np.zeros_like(state)
            for i in range(N):
                x,y,z = state[i]
                ds[i,0] = sigma*(y-x)
                ds[i,1] = x*(rho-z)-y
                ds[i,2] = x*y-beta*z
                # diffusive coupling in x
                xprev = state[(i-1)%N, 0]; xnext = state[(i+1)%N, 0]
                ds[i,0] += coupling*(xprev - 2*x + xnext)
            return ds.flatten()
        x0  = np.random.randn(N*3)
        sol = solve_ivp(ode, [0,t_end], x0, t_eval=np.arange(0,t_end,dt), rtol=1e-7)
        return sol.y.T

    def __repr__(self): return "CoupledOscillators()"
