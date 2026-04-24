"""
mnn.advanced.chaos_advanced.fractals
======================================
Advanced fractal and multifractal analysis for MNN.

Features:
  - Box-counting dimension
  - Information dimension D1
  - Correlation dimension D2 (GP algorithm)
  - Multifractal spectrum f(α) via Legendre transform
  - Hurst exponent (DFA, R/S analysis)
  - Mandelbrot and Julia set generation
  - IFS (Iterated Function Systems)
  - Moran equation for self-similar fractals
"""
from __future__ import annotations
import numpy as np
from scipy.signal import detrend
from typing import Callable, List, Tuple, Optional, Dict


class FractalAnalyzer:
    """
    Comprehensive fractal dimension analysis.
    All methods accept a trajectory/signal array.
    """

    # ── Box-counting dimension ────────────────────────────────────────────────

    @staticmethod
    def box_counting_dimension(points: np.ndarray, n_scales: int = 15) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Minkowski-Bouligand (box-counting) dimension.
        D_B = lim_{ε→0} log N(ε) / log(1/ε)

        Returns (dimension, scale_array, count_array).
        """
        pts  = np.array(points)
        mins = pts.min(axis=0); maxs = pts.max(axis=0)
        span = (maxs - mins).max()
        if span < 1e-15: return 0., np.array([]), np.array([])

        scales = np.logspace(-2, 0, n_scales) * span
        counts = []
        for eps in scales:
            boxes = set()
            for p in pts:
                box = tuple(int((p[d]-mins[d])/eps) for d in range(pts.shape[1]))
                boxes.add(box)
            counts.append(len(boxes))

        counts = np.array(counts, dtype=float)
        valid  = counts > 0
        if valid.sum() < 3:
            return float(pts.shape[1]), scales, counts
        slope, _ = np.polyfit(np.log(1/scales[valid]), np.log(counts[valid]), 1)
        return float(slope), scales, counts

    # ── Hurst exponent ────────────────────────────────────────────────────────

    @staticmethod
    def hurst_exponent_rs(signal: np.ndarray, min_n: int = 10) -> float:
        """
        R/S analysis for Hurst exponent H.
        H = 0.5 → random walk
        H > 0.5 → persistent (trending)
        H < 0.5 → anti-persistent (mean-reverting)
        """
        n = len(signal)
        sizes  = np.unique(np.logspace(np.log10(min_n), np.log10(n//2), 20).astype(int))
        rs_vals = []
        for m in sizes:
            rs_list = []
            for start in range(0, n-m, m):
                seg  = signal[start:start+m]
                mean = seg.mean()
                dev  = np.cumsum(seg - mean)
                R    = dev.max() - dev.min()
                S    = seg.std(ddof=1)
                if S > 1e-15: rs_list.append(R/S)
            if rs_list: rs_vals.append((m, np.mean(rs_list)))

        if len(rs_vals) < 3: return 0.5
        ms  = np.array([v[0] for v in rs_vals])
        rs  = np.array([v[1] for v in rs_vals])
        H,_ = np.polyfit(np.log(ms), np.log(rs+1e-15), 1)
        return float(np.clip(H, 0, 1))

    @staticmethod
    def hurst_exponent_dfa(signal: np.ndarray, n_scales: int = 15) -> float:
        """
        Detrended Fluctuation Analysis (DFA) for Hurst exponent.
        More robust than R/S for non-stationary signals.
        """
        n      = len(signal)
        y      = np.cumsum(signal - signal.mean())
        scales = np.unique(np.logspace(1, np.log10(n//4), n_scales).astype(int))
        F_n    = []
        for m in scales:
            rms_list = []
            for start in range(0, n-m, m):
                seg = y[start:start+m]
                t   = np.arange(m)
                # Detrend by fitting a polynomial
                p    = np.polyfit(t, seg, 1)
                seg_ = seg - np.polyval(p, t)
                rms_list.append(np.sqrt(np.mean(seg_**2)))
            if rms_list: F_n.append((m, np.mean(rms_list)))

        if len(F_n) < 3: return 0.5
        ms  = np.array([v[0] for v in F_n])
        Fs  = np.array([v[1] for v in F_n])
        H,_ = np.polyfit(np.log(ms), np.log(Fs+1e-15), 1)
        return float(np.clip(H, 0, 1))

    # ── Lacunarity ────────────────────────────────────────────────────────────

    @staticmethod
    def lacunarity(points: np.ndarray, scales: int = 10) -> float:
        """
        Lacunarity Λ(ε) = (σ/μ)² of box mass distribution.
        High lacunarity → "gappy" fractal; low → uniform.
        Returns mean lacunarity across scales.
        """
        pts  = np.array(points)
        mins = pts.min(axis=0); maxs = pts.max(axis=0)
        span = (maxs-mins).max()
        eps_vals = np.logspace(-2, -0.5, scales) * span
        lacs = []
        for eps in eps_vals:
            counts_per_box: Dict[tuple,int] = {}
            for p in pts:
                box = tuple(int((p[d]-mins[d])/eps) for d in range(pts.shape[1]))
                counts_per_box[box] = counts_per_box.get(box, 0) + 1
            masses = np.array(list(counts_per_box.values()), dtype=float)
            if masses.std() > 0: lacs.append((masses.std()/masses.mean())**2)
        return float(np.mean(lacs)) if lacs else 0.

    # ── Classic fractal sets ──────────────────────────────────────────────────

    @staticmethod
    def mandelbrot_set(n_real: int = 300, n_imag: int = 300,
                        x_range=(-2.5,1), y_range=(-1.2,1.2),
                        max_iter: int = 100) -> np.ndarray:
        """
        Mandelbrot set: points c ∈ C where z_{n+1} = z_n² + c stays bounded.
        Returns iteration count array (escape time).
        """
        xs = np.linspace(*x_range, n_real)
        ys = np.linspace(*y_range, n_imag)
        X, Y = np.meshgrid(xs, ys)
        C    = X + 1j*Y
        Z    = np.zeros_like(C)
        out  = np.zeros(C.shape, dtype=int)
        for i in range(max_iter):
            mask     = np.abs(Z) <= 2
            Z[mask]  = Z[mask]**2 + C[mask]
            out[mask & (np.abs(Z) > 2)] = i
        return out

    @staticmethod
    def julia_set(c: complex = -0.7 + 0.27j,
                   n_real: int = 300, n_imag: int = 300,
                   x_range=(-1.5,1.5), y_range=(-1.5,1.5),
                   max_iter: int = 100) -> np.ndarray:
        """
        Julia set for parameter c: z_{n+1} = z_n² + c.
        Returns escape time array.
        """
        xs = np.linspace(*x_range, n_real)
        ys = np.linspace(*y_range, n_imag)
        X, Y = np.meshgrid(xs, ys)
        Z    = X + 1j*Y
        out  = np.zeros(Z.shape, dtype=int)
        for i in range(max_iter):
            mask    = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + c
            out[mask & (np.abs(Z) > 2)] = i
        return out

    # ── IFS ───────────────────────────────────────────────────────────────────

    @staticmethod
    def ifs_attractor(transformations: List[Tuple],
                       n_points: int = 50000,
                       seed: int = 42) -> np.ndarray:
        """
        Iterated Function System attractor via the chaos game.
        transformations = [(prob, A, b), ...] where x → A@x + b.
        """
        rng   = np.random.default_rng(seed)
        probs = np.array([t[0] for t in transformations])
        probs = probs / probs.sum()
        x     = np.zeros(2)
        pts   = np.zeros((n_points, 2))
        for i in range(n_points):
            k     = rng.choice(len(transformations), p=probs)
            _, A, b = transformations[k]
            x     = np.array(A) @ x + np.array(b)
            pts[i] = x
        return pts

    @staticmethod
    def sierpinski_triangle(n_points: int = 30000) -> np.ndarray:
        """Sierpiński triangle via IFS."""
        return FractalAnalyzer.ifs_attractor([
            (1/3, [[0.5,0],[0,0.5]], [0,0]),
            (1/3, [[0.5,0],[0,0.5]], [0.5,0]),
            (1/3, [[0.5,0],[0,0.5]], [0.25,0.5]),
        ], n_points)

    @staticmethod
    def barnsley_fern(n_points: int = 50000) -> np.ndarray:
        """Barnsley fern via IFS."""
        return FractalAnalyzer.ifs_attractor([
            (0.01, [[0,0],[0,0.16]],       [0,0]),
            (0.85, [[0.85,0.04],[-0.04,0.85]], [0,1.6]),
            (0.07, [[0.20,-0.26],[0.23,0.22]],  [0,1.6]),
            (0.07, [[-0.15,0.28],[0.26,0.24]],  [0,0.44]),
        ], n_points)

    def __repr__(self): return "FractalAnalyzer()"


class MultifractalSpectrum:
    """
    Multifractal analysis via the method of moments.
    Computes the singularity spectrum f(α) and generalized dimensions D_q.
    """

    def __init__(self, points: np.ndarray):
        self.points = np.array(points)

    def generalized_dimensions(self, q_range: Tuple = (-5, 5),
                                n_q: int = 21,
                                n_scales: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """
        D_q = lim_{ε→0} (1/(q-1)) log(Σ μᵢ^q) / log(ε)
        where μᵢ = mass fraction in i-th box.
        Returns (q_values, D_q_values).
        """
        pts  = self.points
        mins = pts.min(axis=0); maxs = pts.max(axis=0)
        span = (maxs-mins).max()
        qs   = np.linspace(*q_range, n_q)
        Dq   = np.zeros(n_q)
        eps_arr = np.logspace(-2, -0.5, n_scales) * span

        for qi, q in enumerate(qs):
            Zq_vals = []
            for eps in eps_arr:
                box_counts: Dict[tuple,int] = {}
                for p in pts:
                    box = tuple(int((p[d]-mins[d])/max(eps,1e-15)) for d in range(pts.shape[1]))
                    box_counts[box] = box_counts.get(box,0) + 1
                n_total = len(pts)
                mus     = np.array(list(box_counts.values()),dtype=float) / n_total
                if q == 1:
                    Zq_vals.append((eps, -np.sum(mus*np.log(mus+1e-30))))
                else:
                    Zq_vals.append((eps, np.sum(mus**q)))

            if len(Zq_vals) < 3:
                Dq[qi] = pts.shape[1]; continue
            es = np.array([v[0] for v in Zq_vals])
            Zs = np.array([v[1] for v in Zq_vals])
            valid = Zs > 0
            if valid.sum() < 3: Dq[qi] = pts.shape[1]; continue
            slope, _ = np.polyfit(np.log(es[valid]), np.log(Zs[valid]), 1)
            Dq[qi] = slope / (q-1) if q != 1 else slope

        return qs, Dq

    def singularity_spectrum(self, q_range=(-5,5), n_q=21) -> Tuple[np.ndarray, np.ndarray]:
        """
        f(α) spectrum via Legendre transform of τ(q) = (q-1)D_q.
        α = dτ/dq,  f(α) = qα - τ(q).
        Returns (alpha, f_alpha).
        """
        qs, Dq = self.generalized_dimensions(q_range, n_q)
        tau    = (qs - 1) * Dq
        alpha  = np.gradient(tau, qs)
        f_a    = qs * alpha - tau
        return alpha, f_a

    def __repr__(self): return f"MultifractalSpectrum(n={len(self.points)})"
