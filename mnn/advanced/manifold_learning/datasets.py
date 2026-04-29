"""
mnn.advanced.manifold_learning.datasets
=========================================
Manifold dataset generators for MNN.

Generates clean point clouds on exact mathematical manifolds,
with optional noise to simulate real-world data.

Manifolds
---------
  Curves (1D):   S¹ (circle), helix, figure-eight, trefoil knot
  Surfaces (2D): S² (sphere), T² (torus), Klein bottle, Möbius band, Swiss roll
  Higher-dim:    Sⁿ (n-sphere), SO(3), Stiefel manifold, Grassmannian
  Synthetic:     two_moons, swiss_roll, s_curve (classic ML datasets, intrinsically 1-2D)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ManifoldDataset:
    """
    Container for a manifold point cloud dataset.

    Attributes
    ----------
    points         : (N, ambient_dim) array of points in ambient space
    params         : (N, intrinsic_dim) intrinsic parameters (if known)
    manifold_name  : name of the manifold
    ambient_dim    : dimension of the embedding space
    intrinsic_dim  : intrinsic dimension of the manifold
    radius         : characteristic scale (if applicable)
    noise_level    : std of added noise
    """
    points        : np.ndarray
    params        : Optional[np.ndarray]      = None
    manifold_name : str                        = "unknown"
    ambient_dim   : int                        = 3
    intrinsic_dim : int                        = 1
    radius        : float                      = 1.0
    noise_level   : float                      = 0.0
    metadata      : dict                       = field(default_factory=dict)

    def __len__(self):          return len(self.points)
    def __repr__(self):
        return (f"ManifoldDataset({self.manifold_name}, "
                f"n={len(self.points)}, "
                f"ambient={self.ambient_dim}, "
                f"intrinsic={self.intrinsic_dim}, "
                f"noise={self.noise_level})")

    def add_noise(self, std: float) -> "ManifoldDataset":
        """Return a new dataset with Gaussian noise added."""
        noisy = self.points + np.random.randn(*self.points.shape) * std
        return ManifoldDataset(
            points=noisy, params=self.params,
            manifold_name=self.manifold_name + "_noisy",
            ambient_dim=self.ambient_dim, intrinsic_dim=self.intrinsic_dim,
            radius=self.radius, noise_level=std
        )

    def train_test_split(self, test_frac: float = 0.2, seed: int = 0):
        """Split into train/test datasets."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.points))
        n_test = int(len(self.points) * test_frac)
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        def _sub(i):
            return ManifoldDataset(
                points=self.points[i],
                params=self.params[i] if self.params is not None else None,
                manifold_name=self.manifold_name,
                ambient_dim=self.ambient_dim, intrinsic_dim=self.intrinsic_dim,
                radius=self.radius, noise_level=self.noise_level
            )
        return _sub(train_idx), _sub(test_idx)


class ManifoldGenerator:
    """Factory class for generating manifold datasets."""

    # ── Circles and curves ────────────────────────────────────────────────────

    @staticmethod
    def circle(n: int = 1000, radius: float = 1.0,
               noise: float = 0.0, seed: int = 0) -> ManifoldDataset:
        """
        S¹ — unit circle embedded in R².
        Parametrised by θ ∈ [0, 2π):  (r·cos θ, r·sin θ)
        Intrinsic dim = 1, ambient dim = 2.
        """
        rng   = np.random.default_rng(seed)
        theta = rng.uniform(0, 2*np.pi, n)
        x     = radius * np.cos(theta)
        y     = radius * np.sin(theta)
        pts   = np.column_stack([x, y])
        if noise > 0:
            pts += rng.standard_normal((n, 2)) * noise
        return ManifoldDataset(
            points=pts.astype(np.float32),
            params=theta.reshape(-1,1).astype(np.float32),
            manifold_name="S¹", ambient_dim=2, intrinsic_dim=1,
            radius=radius, noise_level=noise
        )

    @staticmethod
    def circle_3d(n: int = 1000, radius: float = 1.0,
                  noise: float = 0.0, seed: int = 0) -> ManifoldDataset:
        """S¹ embedded in R³ (lying in the xy-plane)."""
        rng   = np.random.default_rng(seed)
        theta = rng.uniform(0, 2*np.pi, n)
        pts   = np.column_stack([radius*np.cos(theta),
                                  radius*np.sin(theta),
                                  np.zeros(n)])
        if noise > 0:
            pts += rng.standard_normal((n, 3)) * noise
        return ManifoldDataset(
            points=pts.astype(np.float32),
            params=theta.reshape(-1,1).astype(np.float32),
            manifold_name="S¹⊂R³", ambient_dim=3, intrinsic_dim=1,
            radius=radius, noise_level=noise
        )

    @staticmethod
    def helix(n: int = 1000, radius: float = 1.0, height: float = 4.0,
              turns: int = 3, noise: float = 0.0, seed: int = 0) -> ManifoldDataset:
        """
        Helix in R³:  (r·cos(t), r·sin(t), h·t/(2π·turns))
        Topologically S¹ × R, intrinsic dim = 1.
        """
        rng = np.random.default_rng(seed)
        t   = rng.uniform(0, 2*np.pi*turns, n)
        pts = np.column_stack([
            radius * np.cos(t),
            radius * np.sin(t),
            height * t / (2*np.pi*turns)
        ])
        if noise > 0:
            pts += rng.standard_normal((n, 3)) * noise
        return ManifoldDataset(
            points=pts.astype(np.float32),
            params=t.reshape(-1,1).astype(np.float32),
            manifold_name="Helix", ambient_dim=3, intrinsic_dim=1,
            radius=radius, noise_level=noise
        )

    @staticmethod
    def trefoil_knot(n: int = 1000, noise: float = 0.0, seed: int = 0) -> ManifoldDataset:
        """
        Trefoil knot in R³ — a non-trivial embedding of S¹.
        Parametrised by t ∈ [0, 2π):
          x = sin(t) + 2sin(2t)
          y = cos(t) - 2cos(2t)
          z = -sin(3t)
        """
        rng = np.random.default_rng(seed)
        t   = rng.uniform(0, 2*np.pi, n)
        pts = np.column_stack([
            np.sin(t) + 2*np.sin(2*t),
            np.cos(t) - 2*np.cos(2*t),
            -np.sin(3*t)
        ])
        if noise > 0:
            pts += rng.standard_normal((n, 3)) * noise
        return ManifoldDataset(
            points=pts.astype(np.float32),
            params=t.reshape(-1,1).astype(np.float32),
            manifold_name="TrefoilKnot", ambient_dim=3, intrinsic_dim=1,
            noise_level=noise
        )

    @staticmethod
    def figure_eight(n: int = 1000, noise: float = 0.0, seed: int = 0) -> ManifoldDataset:
        """Lemniscate of Bernoulli (figure-eight) in R²."""
        rng = np.random.default_rng(seed)
        t   = rng.uniform(0, 2*np.pi, n)
        denom = 1 + np.sin(t)**2
        pts = np.column_stack([
            np.sqrt(2) * np.cos(t) / denom,
            np.sqrt(2) * np.sin(t) * np.cos(t) / denom
        ])
        if noise > 0:
            pts += rng.standard_normal((n, 2)) * noise
        return ManifoldDataset(
            points=pts.astype(np.float32),
            params=t.reshape(-1,1).astype(np.float32),
            manifold_name="Figure8", ambient_dim=2, intrinsic_dim=1,
            noise_level=noise
        )

    # ── Surfaces (2D manifolds) ───────────────────────────────────────────────

    @staticmethod
    def sphere(n: int = 2000, radius: float = 1.0,
               noise: float = 0.0, seed: int = 0) -> ManifoldDataset:
        """
        S² — 2-sphere embedded in R³.
        Uniform sampling via Gaussian projection.
        Intrinsic dim = 2, ambient dim = 3.
        """
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((n, 3))
        pts = radius * raw / np.linalg.norm(raw, axis=1, keepdims=True)
        # Intrinsic params: (θ, φ)
        theta = np.arccos(np.clip(pts[:,2]/radius, -1, 1))
        phi   = np.arctan2(pts[:,1], pts[:,0])
        params = np.column_stack([theta, phi])
        if noise > 0:
            pts += rng.standard_normal((n, 3)) * noise
        return ManifoldDataset(
            points=pts.astype(np.float32),
            params=params.astype(np.float32),
            manifold_name="S²", ambient_dim=3, intrinsic_dim=2,
            radius=radius, noise_level=noise
        )

    @staticmethod
    def torus(n: int = 2000, R: float = 2.0, r: float = 0.8,
              noise: float = 0.0, seed: int = 0) -> ManifoldDataset:
        """
        T² — flat torus embedded in R³.
        (u,v) ∈ [0,2π)²:
          x = (R + r·cos v)·cos u
          y = (R + r·cos v)·sin u
          z = r·sin v
        Intrinsic dim = 2.
        """
        rng = np.random.default_rng(seed)
        u   = rng.uniform(0, 2*np.pi, n)
        v   = rng.uniform(0, 2*np.pi, n)
        pts = np.column_stack([
            (R + r*np.cos(v)) * np.cos(u),
            (R + r*np.cos(v)) * np.sin(u),
            r * np.sin(v)
        ])
        if noise > 0:
            pts += rng.standard_normal((n, 3)) * noise
        return ManifoldDataset(
            points=pts.astype(np.float32),
            params=np.column_stack([u, v]).astype(np.float32),
            manifold_name="T²", ambient_dim=3, intrinsic_dim=2,
            radius=R, noise_level=noise,
            metadata={"R": R, "r": r}
        )

    @staticmethod
    def swiss_roll(n: int = 2000, noise: float = 0.1, seed: int = 0) -> ManifoldDataset:
        """
        Swiss roll manifold — a classic 2D manifold in R³.
        Intrinsic dim = 2 (curled rectangle).
        """
        rng = np.random.default_rng(seed)
        t   = 1.5 * np.pi * (1 + 2 * rng.uniform(0, 1, n))
        h   = rng.uniform(0, 10, n)
        pts = np.column_stack([
            t * np.cos(t),
            h,
            t * np.sin(t)
        ])
        if noise > 0:
            pts += rng.standard_normal((n, 3)) * noise
        return ManifoldDataset(
            points=pts.astype(np.float32),
            params=np.column_stack([t, h]).astype(np.float32),
            manifold_name="SwissRoll", ambient_dim=3, intrinsic_dim=2,
            noise_level=noise
        )

    @staticmethod
    def mobius_band(n: int = 2000, noise: float = 0.0, seed: int = 0) -> ManifoldDataset:
        """
        Möbius band embedded in R³ (non-orientable surface).
        u ∈ [0, 2π), v ∈ [-0.5, 0.5]:
          x = (1 + v/2 · cos(u/2)) · cos u
          y = (1 + v/2 · cos(u/2)) · sin u
          z = v/2 · sin(u/2)
        """
        rng = np.random.default_rng(seed)
        u   = rng.uniform(0, 2*np.pi, n)
        v   = rng.uniform(-0.5, 0.5, n)
        pts = np.column_stack([
            (1 + 0.5*v*np.cos(u/2)) * np.cos(u),
            (1 + 0.5*v*np.cos(u/2)) * np.sin(u),
            0.5 * v * np.sin(u/2)
        ])
        if noise > 0:
            pts += rng.standard_normal((n, 3)) * noise
        return ManifoldDataset(
            points=pts.astype(np.float32),
            params=np.column_stack([u, v]).astype(np.float32),
            manifold_name="MöbiusBand", ambient_dim=3, intrinsic_dim=2,
            noise_level=noise
        )

    @staticmethod
    def two_moons(n: int = 1000, noise: float = 0.05, seed: int = 0) -> ManifoldDataset:
        """
        Two interleaved half-circles in R² (classic ML dataset).
        Intrinsic dim = 1.
        """
        rng  = np.random.default_rng(seed)
        n_h  = n // 2
        t1   = np.linspace(0, np.pi, n_h)
        t2   = np.linspace(0, np.pi, n - n_h)
        moon1 = np.column_stack([np.cos(t1), np.sin(t1)])
        moon2 = np.column_stack([1 - np.cos(t2), 1 - np.sin(t2) - 0.5])
        pts   = np.vstack([moon1, moon2])
        labels = np.concatenate([np.zeros(n_h), np.ones(n - n_h)])
        if noise > 0:
            pts += rng.standard_normal((n, 2)) * noise
        return ManifoldDataset(
            points=pts.astype(np.float32),
            params=labels.reshape(-1,1).astype(np.float32),
            manifold_name="TwoMoons", ambient_dim=2, intrinsic_dim=1,
            noise_level=noise
        )

    # ── Higher-dimensional manifolds ──────────────────────────────────────────

    @staticmethod
    def hypersphere(n: int = 2000, ambient_dim: int = 4,
                    radius: float = 1.0, noise: float = 0.0,
                    seed: int = 0) -> ManifoldDataset:
        """
        Sⁿ⁻¹ — (n-1)-sphere embedded in Rⁿ.
        Sampled via Gaussian projection.
        Intrinsic dim = ambient_dim - 1.
        """
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((n, ambient_dim))
        pts = radius * raw / np.linalg.norm(raw, axis=1, keepdims=True)
        if noise > 0:
            pts += rng.standard_normal((n, ambient_dim)) * noise
        return ManifoldDataset(
            points=pts.astype(np.float32),
            params=None,
            manifold_name=f"S^{ambient_dim-1}", ambient_dim=ambient_dim,
            intrinsic_dim=ambient_dim-1, radius=radius, noise_level=noise
        )

    @staticmethod
    def flat_torus_nd(n: int = 2000, dims: int = 4,
                      noise: float = 0.0, seed: int = 0) -> ManifoldDataset:
        """
        Flat n/2-torus T^(d/2) embedded in Rⁿ via:
        (cos θ₁, sin θ₁, cos θ₂, sin θ₂, ...) with dims = 2k.
        Intrinsic dim = dims/2.
        """
        assert dims % 2 == 0, "dims must be even for flat torus"
        rng    = np.random.default_rng(seed)
        d      = dims // 2
        thetas = rng.uniform(0, 2*np.pi, (n, d))
        pts    = np.column_stack([
            col for i in range(d)
            for col in [np.cos(thetas[:,i]), np.sin(thetas[:,i])]
        ])
        if noise > 0:
            pts += rng.standard_normal((n, dims)) * noise
        return ManifoldDataset(
            points=pts.astype(np.float32),
            params=thetas.astype(np.float32),
            manifold_name=f"T^{d}⊂R^{dims}",
            ambient_dim=dims, intrinsic_dim=d, noise_level=noise
        )

    @staticmethod
    def so3_manifold(n: int = 2000, noise: float = 0.0, seed: int = 0) -> ManifoldDataset:
        """
        SO(3) — rotation group, embedded in R^9 (3×3 matrices flattened).
        Sampled via QR decomposition of random Gaussian matrices.
        Intrinsic dim = 3.
        """
        rng = np.random.default_rng(seed)
        pts = []
        for _ in range(n):
            A        = rng.standard_normal((3, 3))
            Q, R     = np.linalg.qr(A)
            Q       *= np.sign(np.diag(R))   # ensure det > 0
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            pts.append(Q.flatten())
        pts = np.array(pts, dtype=np.float32)
        if noise > 0:
            pts += rng.standard_normal((n, 9)).astype(np.float32) * noise
        return ManifoldDataset(
            points=pts, params=None,
            manifold_name="SO(3)", ambient_dim=9, intrinsic_dim=3,
            noise_level=noise
        )

    @staticmethod
    def product_manifold(ds1: ManifoldDataset,
                         ds2: ManifoldDataset,
                         n:   int = None) -> ManifoldDataset:
        """
        Product M₁ × M₂ — Cartesian product of two manifolds.
        If n is given, resample to n points.
        """
        n1 = len(ds1); n2 = len(ds2)
        if n is None:
            n = min(n1, n2)
        idx1 = np.random.choice(n1, n, replace=(n > n1))
        idx2 = np.random.choice(n2, n, replace=(n > n2))
        pts  = np.hstack([ds1.points[idx1], ds2.points[idx2]])
        params = None
        if ds1.params is not None and ds2.params is not None:
            params = np.hstack([ds1.params[idx1], ds2.params[idx2]])
        return ManifoldDataset(
            points=pts.astype(np.float32), params=params,
            manifold_name=f"{ds1.manifold_name}×{ds2.manifold_name}",
            ambient_dim=ds1.ambient_dim + ds2.ambient_dim,
            intrinsic_dim=ds1.intrinsic_dim + ds2.intrinsic_dim
        )

    # ── Convenience all-in-one accessor ──────────────────────────────────────

    @classmethod
    def get(cls, name: str, n: int = 1000, **kwargs) -> ManifoldDataset:
        """
        Convenience factory.  name ∈ {'circle','sphere','torus','helix',
        'trefoil','swiss_roll','mobius','two_moons','hypersphere',
        'figure_eight','so3','flat_torus'}
        """
        dispatch = {
            "circle":      cls.circle,
            "circle_3d":   cls.circle_3d,
            "sphere":      cls.sphere,
            "torus":       cls.torus,
            "helix":       cls.helix,
            "trefoil":     cls.trefoil_knot,
            "figure_eight":cls.figure_eight,
            "swiss_roll":  cls.swiss_roll,
            "mobius":      cls.mobius_band,
            "two_moons":   cls.two_moons,
            "hypersphere": cls.hypersphere,
            "flat_torus":  cls.flat_torus_nd,
            "so3":         cls.so3_manifold,
        }
        if name not in dispatch:
            raise ValueError(f"Unknown manifold '{name}'. Choose from {list(dispatch.keys())}")
        return dispatch[name](n=n, **kwargs)
