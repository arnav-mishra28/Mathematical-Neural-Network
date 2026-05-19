"""mnn.intelligence.discovery — Scientific equation discovery from data.

Pillar 4: Scientific Discovery Engine.
Combines neural networks with sparse regression to automatically
discover governing equations from data. The neural network learns
the dynamics, then sparse regression (SINDy-style) extracts
interpretable symbolic equations.

Example: From Lorenz trajectory data → discovers dx/dt = σ(y-x), etc.
"""
from __future__ import annotations
import torch, torch.nn as nn, numpy as np
from typing import Callable, Optional, Dict, List, Tuple, Union
from tqdm import tqdm
from itertools import combinations_with_replacement

from mnn.neural.base_network import MNNNetwork


class LibraryBuilder:
    """Build a library of candidate functions for sparse regression.

    Given state variables [x₁, x₂, ..., xₙ], constructs a library Θ(X) containing:
      - Constants (1)
      - Linear terms (x₁, x₂, ...)
      - Quadratic terms (x₁², x₁x₂, ...)
      - Cubic terms (optional)
      - Trigonometric terms (sin(xᵢ), cos(xᵢ))
    """
    def __init__(self, state_dim: int, poly_order: int = 2,
                 include_trig: bool = False, include_cross: bool = True,
                 custom_fns: Optional[List[Tuple[str, Callable]]] = None):
        self.state_dim = state_dim
        self.poly_order = poly_order
        self.include_trig = include_trig
        self.include_cross = include_cross
        self.custom_fns = custom_fns or []
        self.feature_names = self._build_names()

    def _build_names(self) -> List[str]:
        names = ["1"]
        var_names = [f"x{i}" for i in range(self.state_dim)]

        # Linear
        names.extend(var_names)

        # Polynomial (order 2+)
        for order in range(2, self.poly_order + 1):
            if self.include_cross:
                for combo in combinations_with_replacement(range(self.state_dim), order):
                    name = "*".join(var_names[i] for i in combo)
                    names.append(name)
            else:
                for i in range(self.state_dim):
                    names.append(f"{var_names[i]}^{order}")

        # Trigonometric
        if self.include_trig:
            for v in var_names:
                names.extend([f"sin({v})", f"cos({v})"])

        # Custom
        for fname, _ in self.custom_fns:
            names.append(fname)

        return names

    def transform(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        features = [np.ones((n, 1))]

        # Linear
        features.append(X)

        # Polynomial
        for order in range(2, self.poly_order + 1):
            if self.include_cross:
                for combo in combinations_with_replacement(range(self.state_dim), order):
                    feat = np.ones((n, 1))
                    for idx in combo:
                        feat = feat * X[:, idx:idx+1]
                    features.append(feat)
            else:
                for i in range(self.state_dim):
                    features.append(X[:, i:i+1] ** order)

        # Trigonometric
        if self.include_trig:
            for i in range(self.state_dim):
                features.append(np.sin(X[:, i:i+1]))
                features.append(np.cos(X[:, i:i+1]))

        # Custom
        for _, fn in self.custom_fns:
            features.append(fn(X).reshape(n, -1))

        return np.hstack(features)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def __repr__(self):
        return f"LibraryBuilder(dim={self.state_dim}, poly={self.poly_order}, n_features={self.n_features})"


class SparseRegressor:
    """Sparse regression (SINDy-style) using Sequential Thresholded Least Squares (STLS).

    Finds sparse coefficient matrix Ξ such that:
      dX/dt ≈ Θ(X) · Ξ

    where Θ(X) is the library matrix and Ξ is sparse.
    """
    def __init__(self, threshold: float = 0.1, max_iter: int = 20,
                 alpha: float = 0.05, normalize: bool = True):
        self.threshold = threshold
        self.max_iter = max_iter
        self.alpha = alpha
        self.normalize = normalize
        self.coefficients: Optional[np.ndarray] = None

    def fit(self, library: np.ndarray, derivatives: np.ndarray) -> np.ndarray:
        n_features = library.shape[1]
        n_targets = derivatives.shape[1]

        if self.normalize:
            norms = np.linalg.norm(library, axis=0, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            Theta_norm = library / norms
        else:
            Theta_norm = library
            norms = np.ones((1, n_features))

        Xi = np.linalg.lstsq(Theta_norm, derivatives, rcond=None)[0]

        for _ in range(self.max_iter):
            small = np.abs(Xi) < self.threshold
            Xi[small] = 0.0

            for j in range(n_targets):
                active = ~small[:, j]
                if np.sum(active) == 0:
                    continue
                Xi[active, j] = np.linalg.lstsq(
                    Theta_norm[:, active], derivatives[:, j], rcond=None)[0]

        self.coefficients = Xi / norms.T
        return self.coefficients

    def predict(self, library: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise ValueError("Must fit first")
        return library @ self.coefficients

    def score(self, library: np.ndarray, derivatives: np.ndarray) -> float:
        pred = self.predict(library)
        ss_res = np.sum((derivatives - pred) ** 2)
        ss_tot = np.sum((derivatives - np.mean(derivatives, axis=0)) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-15))

    def equation_strings(self, feature_names: List[str],
                         var_names: Optional[List[str]] = None) -> List[str]:
        if self.coefficients is None:
            raise ValueError("Must fit first")
        n_targets = self.coefficients.shape[1]
        var_names = var_names or [f"dx{i}/dt" for i in range(n_targets)]
        equations = []

        for j in range(n_targets):
            terms = []
            for i in range(len(feature_names)):
                c = self.coefficients[i, j]
                if abs(c) > 1e-10:
                    if feature_names[i] == "1":
                        terms.append(f"{c:.4f}")
                    else:
                        terms.append(f"{c:.4f}*{feature_names[i]}")
            eq = f"{var_names[j]} = " + (" + ".join(terms) if terms else "0")
            equations.append(eq)

        return equations

    def __repr__(self):
        nnz = int(np.sum(np.abs(self.coefficients) > 1e-10)) if self.coefficients is not None else 0
        return f"SparseRegressor(threshold={self.threshold}, nnz={nnz})"


class NeuralDerivativeEstimator:
    """Estimate time derivatives from noisy trajectory data using a neural network.

    Instead of finite differences (noisy), trains a network to smooth the data
    and then computes derivatives via autograd.
    """
    def __init__(self, state_dim: int, width: int = 64, depth: int = 3,
                 lr: float = 1e-3, device: str = "cpu"):
        self.device = device
        self.state_dim = state_dim
        self.net = MNNNetwork(1, state_dim, width=width, depth=depth).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def _t(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.tensor(np.array(x), dtype=torch.float32).to(self.device)

    def fit(self, t: np.ndarray, trajectory: np.ndarray,
            n_epochs: int = 2000, verbose: bool = False) -> None:
        T = self._t(t.reshape(-1, 1))
        X = self._t(trajectory)
        it = tqdm(range(n_epochs), desc="Smoothing") if verbose else range(n_epochs)

        for ep in it:
            self.optimizer.zero_grad()
            pred = self.net(T)
            loss = nn.functional.mse_loss(pred, X)
            loss.backward()
            self.optimizer.step()

    def compute_derivatives(self, t: np.ndarray) -> np.ndarray:
        self.net.eval()
        T = self._t(t.reshape(-1, 1)).requires_grad_(True)
        X_pred = self.net(T)
        dXdt = []
        for i in range(self.state_dim):
            grad = torch.autograd.grad(X_pred[:, i].sum(), T, create_graph=False)[0]
            dXdt.append(grad.detach().cpu().numpy().flatten())
        return np.column_stack(dXdt)

    def smooth_trajectory(self, t: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            return self.net(self._t(t.reshape(-1, 1))).cpu().numpy()


class HybridDiscovery:
    """Hybrid neural-symbolic equation discovery pipeline.

    Full pipeline:
      1. Neural network smooths noisy trajectory data
      2. Autograd computes clean derivatives
      3. Builds a candidate function library Θ(X)
      4. Sparse regression finds Ξ such that dX/dt ≈ Θ(X)·Ξ
      5. Returns discovered symbolic equations
    """
    def __init__(self, state_dim: int, poly_order: int = 2,
                 include_trig: bool = False, threshold: float = 0.1,
                 device: str = "cpu"):
        self.state_dim = state_dim
        self.library_builder = LibraryBuilder(state_dim, poly_order, include_trig)
        self.sparse_reg = SparseRegressor(threshold=threshold)
        self.deriv_estimator = NeuralDerivativeEstimator(state_dim, device=device)
        self.device = device
        self.discovered_equations: List[str] = []
        self.r2_score: float = 0.0

    def discover(self, t: np.ndarray, trajectory: np.ndarray,
                 n_smooth_epochs: int = 3000, use_finite_diff: bool = False,
                 var_names: Optional[List[str]] = None,
                 verbose: bool = True) -> Dict:
        if verbose:
            print("=== Scientific Discovery Pipeline ===")
            print(f"  State dim: {self.state_dim}")
            print(f"  Library size: {self.library_builder.n_features}")
            print(f"  Data points: {len(t)}")

        # Step 1 & 2: Estimate derivatives
        if use_finite_diff:
            dt = t[1] - t[0]
            derivatives = np.gradient(trajectory, dt, axis=0)
            smoothed = trajectory
            if verbose:
                print("  Derivative method: finite differences")
        else:
            if verbose:
                print("  Step 1: Neural smoothing...")
            self.deriv_estimator.fit(t, trajectory, n_epochs=n_smooth_epochs, verbose=verbose)
            smoothed = self.deriv_estimator.smooth_trajectory(t)
            if verbose:
                print("  Step 2: Computing derivatives via autograd...")
            derivatives = self.deriv_estimator.compute_derivatives(t)

        # Step 3: Build library
        if verbose:
            print("  Step 3: Building candidate library...")
        library = self.library_builder.transform(smoothed)

        # Step 4: Sparse regression
        if verbose:
            print("  Step 4: Sparse regression...")
        self.sparse_reg.fit(library, derivatives)
        self.r2_score = self.sparse_reg.score(library, derivatives)

        # Step 5: Extract equations
        deriv_names = var_names or [f"dx{i}/dt" for i in range(self.state_dim)]
        self.discovered_equations = self.sparse_reg.equation_strings(
            self.library_builder.feature_names, deriv_names)

        if verbose:
            print(f"\n  R² Score: {self.r2_score:.6f}")
            print("  Discovered Equations:")
            for eq in self.discovered_equations:
                print(f"    {eq}")

        return {
            "equations": self.discovered_equations,
            "coefficients": self.sparse_reg.coefficients,
            "r2_score": self.r2_score,
            "feature_names": self.library_builder.feature_names,
            "library": library,
            "derivatives": derivatives,
            "smoothed_trajectory": smoothed,
        }

    def predict_dynamics(self, X: np.ndarray) -> np.ndarray:
        library = self.library_builder.transform(X)
        return self.sparse_reg.predict(library)

    def __repr__(self):
        return (f"HybridDiscovery(dim={self.state_dim}, "
                f"features={self.library_builder.n_features}, "
                f"r2={self.r2_score:.4f})")


class ScientificDiscoveryEngine:
    """Top-level discovery engine that orchestrates multiple discovery attempts.

    Supports:
      - Multi-threshold sweeps to find optimal sparsity
      - Cross-validation of discovered equations
      - Comparison of polynomial vs trigonometric libraries
      - Automatic complexity-accuracy tradeoff selection
    """
    def __init__(self, state_dim: int, device: str = "cpu"):
        self.state_dim = state_dim
        self.device = device
        self.best_discovery: Optional[Dict] = None
        self.all_results: List[Dict] = []

    def auto_discover(self, t: np.ndarray, trajectory: np.ndarray,
                      thresholds: Optional[List[float]] = None,
                      poly_orders: Optional[List[int]] = None,
                      try_trig: bool = True,
                      var_names: Optional[List[str]] = None,
                      verbose: bool = True) -> Dict:
        thresholds = thresholds or [0.01, 0.05, 0.1, 0.2, 0.5]
        poly_orders = poly_orders or [2, 3]
        configs = []

        for po in poly_orders:
            configs.append({"poly_order": po, "include_trig": False})
            if try_trig:
                configs.append({"poly_order": po, "include_trig": True})

        best_score = -np.inf
        best_result = None

        for config in configs:
            for thresh in thresholds:
                hd = HybridDiscovery(
                    self.state_dim,
                    poly_order=config["poly_order"],
                    include_trig=config["include_trig"],
                    threshold=thresh,
                    device=self.device,
                )
                try:
                    result = hd.discover(t, trajectory, var_names=var_names,
                                          verbose=False)
                    nnz = int(np.sum(np.abs(result["coefficients"]) > 1e-10))
                    # Penalize complexity: score = R² - λ·(nnz / n_features)
                    complexity_penalty = 0.05 * nnz / hd.library_builder.n_features
                    adjusted_score = result["r2_score"] - complexity_penalty

                    record = {
                        "config": config, "threshold": thresh,
                        "r2": result["r2_score"], "adjusted_score": adjusted_score,
                        "nnz": nnz, "equations": result["equations"],
                        "coefficients": result["coefficients"],
                    }
                    self.all_results.append(record)

                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_result = record

                except Exception:
                    continue

        self.best_discovery = best_result

        if verbose and best_result:
            print("\n=== Best Discovery ===")
            print(f"  Config: poly={best_result['config']['poly_order']}, "
                  f"trig={best_result['config']['include_trig']}")
            print(f"  Threshold: {best_result['threshold']}")
            print(f"  R²: {best_result['r2']:.6f}")
            print(f"  Non-zero terms: {best_result['nnz']}")
            print("  Equations:")
            for eq in best_result["equations"]:
                print(f"    {eq}")

        return best_result or {}

    def cross_validate(self, t: np.ndarray, trajectory: np.ndarray,
                       n_folds: int = 5, **kwargs) -> Dict:
        n = len(t)
        fold_size = n // n_folds
        scores = []

        for fold in range(n_folds):
            test_idx = slice(fold * fold_size, (fold + 1) * fold_size)
            train_idx = np.concatenate([
                np.arange(0, fold * fold_size),
                np.arange((fold + 1) * fold_size, n)
            ]).astype(int)

            hd = HybridDiscovery(self.state_dim, device=self.device, **kwargs)
            hd.discover(t[train_idx], trajectory[train_idx], verbose=False)

            test_lib = hd.library_builder.transform(trajectory[test_idx])
            test_deriv = np.gradient(trajectory[test_idx], t[1] - t[0], axis=0)
            scores.append(hd.sparse_reg.score(test_lib, test_deriv))

        return {"mean_r2": float(np.mean(scores)), "std_r2": float(np.std(scores)),
                "fold_scores": scores}

    def __repr__(self):
        n = len(self.all_results)
        best = f", best_r2={self.best_discovery['r2']:.4f}" if self.best_discovery else ""
        return f"ScientificDiscoveryEngine(dim={self.state_dim}, n_trials={n}{best})"
