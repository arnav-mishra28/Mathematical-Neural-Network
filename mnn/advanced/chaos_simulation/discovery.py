"""
mnn.advanced.chaos_simulation.discovery
==========================================
Equation Discovery Engine — Neural → Symbolic.

Implements SINDy (Sparse Identification of Nonlinear Dynamics)
enhanced with neural network feature learning.

Core idea:
  dx/dt ≈ Θ(x) · ξ
where:
  Θ(x) = library of candidate functions [1, x, y, z, xy, xz, x², ...]
  ξ     = sparse coefficient matrix (discovered via LASSO/thresholding)

Extensions in MNN:
  1. Standard SINDy with polynomial library
  2. Neural-SINDy: learn the library basis with a network
  3. Sparse autoencoder: discover intrinsic coordinates
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from .simulator import ChaosTrajectory


@dataclass
class SymbolicTerm:
    """A single term in the discovered equation library."""
    name:        str
    coefficient: float
    is_active:   bool   = True   # True if |coefficient| > threshold

    def __repr__(self):
        sign = "+" if self.coefficient >= 0 else "-"
        return f"{sign}{abs(self.coefficient):.4f}·{self.name}"


@dataclass
class DiscoveredEquation:
    """One discovered ODE equation: dx_i/dt = Σ ξ_j Θ_j(x)"""
    variable:    str
    terms:       List[SymbolicTerm]
    r2_score:    float = 0.0

    @property
    def active_terms(self) -> List[SymbolicTerm]:
        return [t for t in self.terms if t.is_active]

    def to_string(self) -> str:
        active = self.active_terms
        if not active: return f"d{self.variable}/dt = 0"
        parts = [f"{t.coefficient:+.4f}·{t.name}" for t in active]
        return f"d{self.variable}/dt = " + " ".join(parts)

    def __repr__(self): return self.to_string()


class SINDyEngine:
    """
    SINDy (Sparse Identification of Nonlinear Dynamics) implementation.

    Algorithm:
      1. Build feature library Θ(X) from trajectory states
      2. Solve sparse regression: Ẋ = Θ(X) · Ξ
      3. Threshold small coefficients (sparsity promotion)
      4. Report discovered equations

    Reference: Brunton, Proctor, Kutz (2016), PNAS.
    """

    def __init__(self, poly_degree: int = 2,
                 include_trig: bool = False,
                 include_products: bool = True,
                 threshold: float = 0.05,
                 max_iter: int = 10):
        self.poly_degree      = poly_degree
        self.include_trig     = include_trig
        self.include_products = include_products
        self.threshold        = threshold
        self.max_iter         = max_iter
        self._feature_names: List[str] = []

    # ── Feature library ───────────────────────────────────────

    def build_library(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Build the candidate function library Θ(X).
        X: (N, n) state matrix.
        Returns: (N, n_features) library matrix + feature names.
        """
        N, n    = X.shape
        names   = []
        columns = []

        # Constant
        columns.append(np.ones((N, 1)))
        names.append("1")

        # Linear terms: x₀, x₁, ..., xₙ₋₁
        for i in range(n):
            columns.append(X[:, i:i+1])
            names.append(f"x{i}")

        # Polynomial terms up to degree poly_degree
        if self.poly_degree >= 2:
            for i in range(n):
                for j in range(i, n):
                    columns.append((X[:, i] * X[:, j]).reshape(-1, 1))
                    names.append(f"x{i}·x{j}")

        if self.poly_degree >= 3:
            for i in range(n):
                columns.append((X[:, i]**3).reshape(-1, 1))
                names.append(f"x{i}³")
            for i in range(n):
                for j in range(i, n):
                    for k in range(j, n):
                        columns.append((X[:,i]*X[:,j]*X[:,k]).reshape(-1,1))
                        names.append(f"x{i}·x{j}·x{k}")

        # Trigonometric (optional)
        if self.include_trig:
            for i in range(n):
                columns.append(np.sin(X[:, i:i+1]))
                names.append(f"sin(x{i})")
                columns.append(np.cos(X[:, i:i+1]))
                names.append(f"cos(x{i})")

        Theta = np.hstack(columns)
        self._feature_names = names
        return Theta, names

    # ── Sparse regression ─────────────────────────────────────

    def _sequential_threshold_least_squares(self, Theta: np.ndarray,
                                              dX: np.ndarray) -> np.ndarray:
        """
        Sequential Thresholded Least Squares (STLS) — SINDy's sparse solver.
        Iteratively solves LS and zeroes out small coefficients.
        """
        n_features = Theta.shape[1]
        n_vars     = dX.shape[1]
        Xi         = np.linalg.lstsq(Theta, dX, rcond=None)[0]  # initial LS solve

        for _ in range(self.max_iter):
            # Threshold: zero out small coefficients
            small = np.abs(Xi) < self.threshold
            Xi[small] = 0.0
            # Re-solve on active terms for each variable
            for k in range(n_vars):
                active = ~small[:, k]
                if active.sum() == 0:
                    Xi[:, k] = 0.0
                    continue
                try:
                    Xi[active, k] = np.linalg.lstsq(
                        Theta[:, active], dX[:, k], rcond=None
                    )[0]
                except np.linalg.LinAlgError:
                    pass
        return Xi

    def fit(self, trajectory: ChaosTrajectory,
            var_names: Optional[List[str]] = None) -> List[DiscoveredEquation]:
        """
        Discover equations from a trajectory.

        Parameters
        ----------
        trajectory : ChaosTrajectory with states + derivatives
        var_names  : names for state variables (e.g. ["x","y","z"])

        Returns
        -------
        List of DiscoveredEquation objects (one per state variable).
        """
        X    = trajectory.states.astype(np.float64)     # (N, dim)
        dX   = trajectory.derivatives.astype(np.float64) # (N, dim)
        n    = X.shape[1]
        vnames = var_names or [f"x{i}" for i in range(n)]

        # Build library
        Theta, feat_names = self.build_library(X)

        # Sparse regression
        Xi = self._sequential_threshold_least_squares(Theta, dX)

        # Build equation objects
        equations = []
        for k, vname in enumerate(vnames):
            terms = []
            for j, fname in enumerate(feat_names):
                coeff  = float(Xi[j, k])
                active = abs(coeff) >= self.threshold
                terms.append(SymbolicTerm(fname, coeff, active))
            # R² score
            dX_pred  = Theta @ Xi[:, k]
            ss_res   = np.sum((dX[:, k] - dX_pred)**2)
            ss_tot   = np.sum((dX[:, k] - dX[:, k].mean())**2) + 1e-15
            r2       = float(1 - ss_res / ss_tot)
            equations.append(DiscoveredEquation(vname, terms, r2))

        return equations

    def print_equations(self, equations: List[DiscoveredEquation]):
        print("\n  ╔══════════════════════════════════════════════╗")
        print("  ║     SINDy Discovered Equations               ║")
        print("  ╠══════════════════════════════════════════════╣")
        for eq in equations:
            print(f"  ║  {eq.to_string():<45}║")
            print(f"  ║    R² = {eq.r2_score:.4f}{'':<37}║")
        print("  ╚══════════════════════════════════════════════╝")


class EquationDiscovery:
    """
    Neural-enhanced equation discovery.

    Combines:
    1. SINDy for sparse linear identification
    2. Neural feature learning for nonlinear basis discovery
    3. Sparsity-regularised neural network (Neural-SINDy)
    """

    def __init__(self, state_dim: int, library_dim: int = None,
                 hidden_width: int = 64, threshold: float = 0.05):
        self.state_dim   = state_dim
        self.threshold   = threshold
        self.sindy       = SINDyEngine(poly_degree=2, threshold=threshold)
        lib_dim = library_dim or (1 + state_dim + state_dim*(state_dim+1)//2)
        self.lib_dim     = lib_dim

        # Neural feature encoder: x → φ(x) (learned basis)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_width), nn.Tanh(),
            nn.Linear(hidden_width, hidden_width), nn.Tanh(),
            nn.Linear(hidden_width, lib_dim)
        )
        # Sparse coefficient matrix (learnable)
        self.Xi = nn.Parameter(torch.randn(lib_dim, state_dim) * 0.1)

        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, 0.5); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict dx/dt = Φ(x) · Ξ"""
        phi = self.encoder(x)     # (N, lib_dim)
        return phi @ self.Xi      # (N, state_dim)

    def sparsity_loss(self, lambda_sparse: float = 0.001) -> torch.Tensor:
        """L1 regularisation on Xi to promote sparsity."""
        return lambda_sparse * torch.sum(torch.abs(self.Xi))

    def threshold_Xi(self):
        """Zero out coefficients below threshold (hard thresholding)."""
        with torch.no_grad():
            self.Xi[torch.abs(self.Xi) < self.threshold] = 0.0

    def train_neural_sindy(self, trajectory: ChaosTrajectory,
                            n_epochs: int = 2000, lr: float = 1e-3,
                            lambda_sparse: float = 0.001,
                            verbose: bool = True,
                            print_every: int = 400) -> Dict:
        """Train neural SINDy: learn basis + sparse coefficients simultaneously."""
        params = list(self.encoder.parameters()) + [self.Xi]
        opt    = torch.optim.Adam(params, lr=lr)
        sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

        X  = torch.tensor(trajectory.states,      dtype=torch.float32)
        dX = torch.tensor(trajectory.derivatives, dtype=torch.float32)
        N  = X.shape[0]
        history = {"recon":[], "sparse":[], "total":[]}

        from tqdm import tqdm
        itr = tqdm(range(n_epochs), desc="Neural SINDy") if verbose else range(n_epochs)

        for ep in itr:
            idx    = torch.randperm(N)[:min(512, N)]
            x_b    = X[idx]; dx_b = dX[idx]
            opt.zero_grad()
            dx_pred = self.forward(x_b)
            recon   = nn.functional.mse_loss(dx_pred, dx_b)
            sparse  = self.sparsity_loss(lambda_sparse)
            total   = recon + sparse
            total.backward(); opt.step(); sched.step()

            history["recon"].append(float(recon.detach()))
            history["sparse"].append(float(sparse.detach()))
            history["total"].append(float(total.detach()))

            if verbose and (ep+1) % print_every == 0:
                tqdm.write(f"  [{ep+1:>5}]  recon={float(recon.detach()):.5f}  sparse={float(sparse.detach()):.5f}")

        # Hard threshold after training
        self.threshold_Xi()
        return history

    def get_active_coefficients(self, var_names: Optional[List[str]] = None) -> Dict:
        """Return the active (non-zero) coefficients."""
        xi   = self.Xi.detach().numpy()
        n    = self.state_dim
        vnames = var_names or [f"x{i}" for i in range(n)]
        result = {}
        for k, vname in enumerate(vnames):
            active = {f"phi_{j}": float(xi[j, k])
                      for j in range(self.lib_dim)
                      if abs(xi[j, k]) >= self.threshold}
            result[f"d{vname}/dt"] = active
        return result

    def run_classical_sindy(self, trajectory: ChaosTrajectory,
                             var_names: Optional[List[str]] = None) -> List[DiscoveredEquation]:
        """Run classical SINDy on the trajectory."""
        return self.sindy.fit(trajectory, var_names)
