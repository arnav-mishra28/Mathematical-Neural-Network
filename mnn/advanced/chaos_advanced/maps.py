"""
mnn.advanced.chaos_advanced.maps
==================================
Advanced chaotic maps for MNN Phase 2.

Covers:
  - Standard map (Chirikov-Taylor)
  - Arnold's cat map
  - Tent map
  - Hénon map
  - Ikeda map
  - Bernoulli shift
  - Circle map (quasiperiodicity → chaos route)
  - Kicked rotor
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


class ChaoticMap:
    """Collection of 2D and 1D chaotic maps."""

    # ── 2D area-preserving maps ───────────────────────────────────────────────

    @staticmethod
    def standard_map(theta0: float = 0.3, p0: float = 0.3,
                      K: float = 1.0, n_iter: int = 5000) -> np.ndarray:
        """
        Chirikov-Taylor standard map (area-preserving):
        p_{n+1} = p_n + K sin(θ_n)  mod 2π
        θ_{n+1} = θ_n + p_{n+1}     mod 2π
        K < 0.97 → KAM tori; K > 0.97 → global chaos.
        """
        theta, p = theta0, p0
        pts = np.zeros((n_iter, 2))
        for i in range(n_iter):
            p     = (p + K*np.sin(theta)) % (2*np.pi)
            theta = (theta + p) % (2*np.pi)
            pts[i] = [theta, p]
        return pts

    @staticmethod
    def standard_map_phase_portrait(K: float = 1.0,
                                     n_orbits: int = 20,
                                     n_iter: int = 3000) -> np.ndarray:
        """Multiple orbits of the standard map."""
        all_pts = []
        rng = np.random.default_rng(0)
        for _ in range(n_orbits):
            theta0 = rng.uniform(0, 2*np.pi)
            p0     = rng.uniform(0, 2*np.pi)
            pts    = ChaoticMap.standard_map(theta0, p0, K, n_iter)
            all_pts.append(pts)
        return np.vstack(all_pts)

    @staticmethod
    def arnolds_cat_map(x0: float = 0.1, y0: float = 0.4,
                         n_iter: int = 100) -> np.ndarray:
        """
        Arnold's cat map on T² = [0,1)²:
        [x'] = [1 1] [x]  mod 1
        [y']   [1 2] [y]
        Hyperbolic, mixing, exact. Returns to initial position after finite steps.
        """
        A   = np.array([[1,1],[1,2]])
        x   = np.array([x0, y0])
        pts = np.zeros((n_iter, 2))
        for i in range(n_iter):
            x      = A @ x % 1.0
            pts[i] = x
        return pts

    @staticmethod
    def ikeda_map(x0: float = 0.1, y0: float = 0.1,
                   u: float = 0.9, n_iter: int = 5000) -> np.ndarray:
        """
        Ikeda map (laser cavity model):
        t_{n+1} = 0.4 - 6/(1+x²+y²)
        x_{n+1} = 1 + u(x cos t - y sin t)
        y_{n+1} =   u(x sin t + y cos t)
        u > 0.6 → chaos.
        """
        x, y = x0, y0
        pts  = np.zeros((n_iter, 2))
        for i in range(n_iter):
            t = 0.4 - 6/(1 + x**2 + y**2)
            xn = 1 + u*(x*np.cos(t) - y*np.sin(t))
            yn =     u*(x*np.sin(t) + y*np.cos(t))
            x, y = xn, yn
            pts[i] = [x, y]
        return pts

    @staticmethod
    def henon_map(x0: float = 0., y0: float = 0.,
                   a: float = 1.4, b: float = 0.3,
                   n_iter: int = 10000) -> np.ndarray:
        """
        Hénon map: x_{n+1} = 1 − ax² + y,  y_{n+1} = bx.
        Classic parameters: a=1.4, b=0.3 → strange attractor.
        """
        x, y = x0, y0
        pts  = np.zeros((n_iter, 2))
        for i in range(n_iter):
            xn = 1 - a*x**2 + y
            yn = b*x
            x, y = xn, yn
            pts[i] = [x, y]
        return pts

    # ── 1D maps ───────────────────────────────────────────────────────────────

    @staticmethod
    def tent_map(x0: float = 0.3, r: float = 2.0,
                  n_iter: int = 5000) -> np.ndarray:
        """
        Tent map: x_{n+1} = r·min(x, 1-x).
        r=2: topologically conjugate to the logistic map at r=4.
        """
        x   = x0
        pts = np.zeros(n_iter)
        for i in range(n_iter):
            x      = r * min(x, 1-x)
            pts[i] = x
        return pts

    @staticmethod
    def bernoulli_shift(x0: float = 0.3, n_iter: int = 5000) -> np.ndarray:
        """
        Bernoulli shift: x_{n+1} = 2x mod 1.
        Exact, entropy = log 2, Lyapunov = log 2.
        """
        x   = x0
        pts = np.zeros(n_iter)
        for i in range(n_iter):
            x = (2*x) % 1.0; pts[i] = x
        return pts

    @staticmethod
    def circle_map(theta0: float = 0.1,
                   omega: float = 0.5,
                   K: float = 1.5,
                   n_iter: int = 5000) -> np.ndarray:
        """
        Circle map (Arnold tongue): θ_{n+1} = θ_n + ω − (K/2π)sin(2πθ_n) mod 1.
        K=0: quasiperiodic. K→1: period doubling. K>1: chaotic.
        """
        theta = theta0; pts = np.zeros(n_iter)
        for i in range(n_iter):
            theta = (theta + omega - (K/(2*np.pi))*np.sin(2*np.pi*theta)) % 1.0
            pts[i] = theta
        return pts

    # ── Kicked rotor ──────────────────────────────────────────────────────────

    @staticmethod
    def kicked_rotor(theta0: float = 1.0, p0: float = 0.0,
                      K: float = 5.0, n_iter: int = 5000) -> np.ndarray:
        """
        Quantum/classical kicked rotor (same as standard map in classical limit).
        Returns (angle, momentum) trajectory.
        """
        return ChaoticMap.standard_map(theta0, p0, K, n_iter)

    # ── Return maps ───────────────────────────────────────────────────────────

    @staticmethod
    def lorenz_return_map(trajectory: np.ndarray,
                           var_idx: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lorenz return map: plot z_max(n+1) vs z_max(n).
        Reveals tent-map structure of the Lorenz attractor.
        """
        z    = trajectory[:, var_idx]
        maxima_idx = [i for i in range(1,len(z)-1) if z[i]>z[i-1] and z[i]>z[i+1]]
        zmax = z[maxima_idx]
        return zmax[:-1], zmax[1:]

    def __repr__(self): return "ChaoticMap()"
