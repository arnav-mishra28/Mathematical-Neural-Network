"""mnn.quantum_pde.pde_discovery — PDE Discovery Engine.

Part 7: Given observed dynamics, infer the governing PDE.
Discovers equations like ∂u/∂t = α∇²u from data using
sparse regression on a library of differential operators.
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple


class PDELibrary:
    """Library of candidate differential terms for PDE discovery.

    Given u(x,t), builds candidates: u, u_x, u_xx, u*u_x, u^2, etc.
    """
    def __init__(self, poly_order: int = 3, deriv_order: int = 3,
                 include_products: bool = True):
        self.poly_order = poly_order
        self.deriv_order = deriv_order
        self.include_products = include_products

    def build(self, u: np.ndarray, dx: float, dt: float) -> Tuple[np.ndarray, List[str]]:
        """Build library matrix Θ and labels.

        u: (n_time, n_space) solution data.
        Returns: Θ (n_time*n_space, n_terms), labels
        """
        nt, nx = u.shape
        terms = []
        labels = []

        # Spatial derivatives up to deriv_order
        derivs = [u]
        deriv_labels = ["u"]
        for d in range(1, self.deriv_order + 1):
            du = self._spatial_derivative(u, dx, d)
            derivs.append(du)
            suffix = "x" * d
            deriv_labels.append(f"u_{suffix}")

        # Polynomial terms
        for p in range(self.poly_order + 1):
            for d_idx, (deriv, dlabel) in enumerate(zip(derivs, deriv_labels)):
                if p == 0 and d_idx == 0:
                    terms.append(np.ones_like(u))
                    labels.append("1")
                elif p == 1:
                    terms.append(deriv)
                    labels.append(dlabel)
                elif p > 1 and d_idx == 0:
                    terms.append(u**p)
                    labels.append(f"u^{p}")

        # Product terms: u * u_x, u * u_xx, etc.
        if self.include_products:
            for d_idx in range(1, len(derivs)):
                terms.append(u * derivs[d_idx])
                labels.append(f"u*{deriv_labels[d_idx]}")
                if self.poly_order >= 2:
                    terms.append(u**2 * derivs[d_idx])
                    labels.append(f"u^2*{deriv_labels[d_idx]}")

        # Flatten: (nt, nx) -> (nt*nx,) for each term
        n_interior = (nt - 2) * (nx - 4)  # skip boundaries
        Theta = np.zeros((n_interior, len(terms)))
        for j, term in enumerate(terms):
            Theta[:, j] = term[1:-1, 2:-2].flatten()

        return Theta, labels

    def _spatial_derivative(self, u: np.ndarray, dx: float, order: int) -> np.ndarray:
        """Finite difference spatial derivative."""
        result = u.copy()
        for _ in range(order):
            result = np.gradient(result, dx, axis=1)
        return result


class SparsePDERegressor:
    """Sparse regression to discover PDE coefficients.

    Solves: u_t = Θ ξ with sparsity constraints (sequential thresholding).
    """
    def __init__(self, threshold: float = 0.05, max_iter: int = 20,
                 alpha: float = 0.01):
        self.threshold = threshold
        self.max_iter = max_iter
        self.alpha = alpha

    def fit(self, Theta: np.ndarray, u_t: np.ndarray) -> np.ndarray:
        """Sequential thresholded least squares (STRidge).

        Theta: (n_samples, n_terms) library matrix
        u_t: (n_samples,) time derivative
        Returns: xi (n_terms,) sparse coefficient vector
        """
        # Initial ridge regression
        n = Theta.shape[1]
        xi = np.linalg.lstsq(
            Theta.T @ Theta + self.alpha * np.eye(n),
            Theta.T @ u_t, rcond=None)[0]

        for _ in range(self.max_iter):
            small = np.abs(xi) < self.threshold
            xi[small] = 0
            big = ~small
            if np.sum(big) == 0:
                break
            xi[big] = np.linalg.lstsq(
                Theta[:, big].T @ Theta[:, big] + self.alpha * np.eye(int(np.sum(big))),
                Theta[:, big].T @ u_t, rcond=None)[0]

        return xi


class PDEDiscoveryEngine:
    """Discover governing PDE from observed spatiotemporal data.

    Pipeline: compute u_t → build library Θ → sparse regression → identify PDE.
    """
    def __init__(self, poly_order: int = 3, deriv_order: int = 3,
                 threshold: float = 0.05):
        self.library = PDELibrary(poly_order, deriv_order)
        self.regressor = SparsePDERegressor(threshold)

    def discover(self, u: np.ndarray, dx: float, dt: float,
                  verbose: bool = False) -> Dict:
        """Discover PDE from solution data u(x,t).

        u: (n_time, n_space) array.
        Returns dict with discovered equation and coefficients.
        """
        nt, nx = u.shape

        # Compute time derivative
        u_t = np.gradient(u, dt, axis=0)
        u_t_interior = u_t[1:-1, 2:-2].flatten()

        # Build library
        Theta, labels = self.library.build(u, dx, dt)

        # Sparse regression
        xi = self.regressor.fit(Theta, u_t_interior)

        # Build equation string
        active = np.where(np.abs(xi) > 1e-10)[0]
        terms = []
        for idx in active:
            coeff = xi[idx]
            label = labels[idx]
            if abs(coeff) > 1e-10:
                terms.append(f"{coeff:+.4f}*{label}")

        equation = "u_t = " + " ".join(terms) if terms else "u_t = 0"
        residual = np.linalg.norm(Theta @ xi - u_t_interior) / (np.linalg.norm(u_t_interior) + 1e-15)

        if verbose:
            print(f"  Discovered: {equation}")
            print(f"  Active terms: {len(active)}/{len(labels)}")
            print(f"  Relative residual: {residual:.6f}")
            for idx in active:
                print(f"    {labels[idx]}: {xi[idx]:.6f}")

        return {
            "equation": equation,
            "coefficients": xi,
            "labels": labels,
            "active_terms": [labels[i] for i in active],
            "active_coeffs": {labels[i]: float(xi[i]) for i in active},
            "residual": float(residual),
            "n_terms": len(active),
        }
